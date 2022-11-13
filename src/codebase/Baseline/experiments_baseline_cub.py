import logging
import os
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
import pickle
import dataset.utils_dataset as utils_dataset
import utils
from BB.models.VIT import VisionTransformer_baseline, CONFIGS
from Baseline.models.baseline_cub import BaseLine_CUB
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Logger.logger_cubs import Logger_CUBS

logger = logging.getLogger(__name__)


def test_explainer(args):
    print("###############################################")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", root, "explainer")
    output_path = os.path.join(args.output, args.dataset, "Baseline", root, "explainer")
    tb_logs_path = os.path.join(args.logs, args.dataset, "Baseline", f"{root}_explainer")

    print("########### Paths ###########")
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    print("########### Paths ###########")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")
    pickle.dump(args, open(os.path.join(output_path, "test_explainer_configs.pkl"), "wb"))
    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["save_transform"]
    val_transform = transforms["save_transform"]
    val_loader = utils_dataset.get_test_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        val_transform,
    )

    if args.arch == "ResNet101":
        baseline = BaseLine_CUB(args).to(device)
        baseline.load_state_dict(torch.load(os.path.join(args.bb_chkpt, args.bb_chkpt_file)))
    else:
        baseline = VisionTransformer_baseline(
            CONFIGS[args.arch], args.img_size, zero_head=True, num_classes=len(args.labels),
            op_size=len(args.concept_names), smoothing_value=args.smoothing_value
        ).to(device)
        baseline.load_state_dict(torch.load(os.path.join(args.bb_chkpt, args.bb_chkpt_file))["model"])

    args.input_size_pi = 2048

    cur_glt_chkpt = os.path.join(chk_pt_path, "g_best_model_epoch_77.pth.tar")
    g = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(device)
    g.load_state_dict(torch.load(cur_glt_chkpt))
    g.eval()

    predict(
        args.epochs_g,
        baseline,
        g,
        val_loader,
        logger,
        args.dataset,
        device,
        output_path,
        mode="test"
    )


def predict(
        epochs_g,
        baseline,
        g,
        val_loader,
        logger,
        dataset,
        device,
        output_path,
        mode
):
    tensor_images = torch.FloatTensor()
    tensor_concepts = torch.FloatTensor().cuda()
    tensor_preds = torch.FloatTensor().cuda()
    tensor_y = torch.FloatTensor().cuda()
    tensor_conceptizator_concepts = torch.FloatTensor().cuda()
    # tensor_conceptizator_threshold = torch.FloatTensor().cuda()
    tensor_concept_mask = torch.FloatTensor().cuda()
    tensor_alpha = torch.FloatTensor().cuda()
    tensor_alpha_norm = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as t:
            for batch_id, (image, label, attributes) in enumerate(val_loader):
                image, label = image.to(device), label.to(torch.long).to(device)
                logits_concepts = baseline(image)
                y_hat, selection_out, auxiliary_out, concept_mask, \
                alpha, alpha_norm, \
                conceptizator = g(logits_concepts, test=True)

                t.set_postfix(epoch='{0}'.format(batch_id))
                t.update()

                tensor_images = torch.cat((tensor_images, image.cpu()), dim=0)
                tensor_concepts = torch.cat((tensor_concepts, logits_concepts), dim=0)
                tensor_preds = torch.cat((tensor_preds, y_hat), dim=0)

                tensor_y = torch.cat((tensor_y, label), dim=0)
                tensor_conceptizator_concepts = torch.cat(
                    (tensor_conceptizator_concepts, conceptizator.concepts), dim=1
                )
                tensor_concept_mask = concept_mask
                tensor_alpha = alpha
                tensor_alpha_norm = alpha_norm

    tensor_concepts = tensor_concepts.cpu()
    tensor_preds = tensor_preds.cpu()

    tensor_y = tensor_y.cpu()

    tensor_conceptizator_concepts = tensor_conceptizator_concepts.cpu()
    # tensor_conceptizator_threshold = tensor_conceptizator_threshold.cpu()
    tensor_concept_mask = tensor_concept_mask.cpu()
    tensor_alpha = tensor_alpha.cpu()
    tensor_alpha_norm = tensor_alpha_norm.cpu()

    print("Output sizes: ")
    print(f"tensor_images size: {tensor_images.size()}")
    print(f"tensor_concepts size: {tensor_concepts.size()}")
    print(f"tensor_preds size: {tensor_preds.size()}")
    print(f"tensor_y size: {tensor_y.size()}")
    print(f"tensor_conceptizator_concepts size: {tensor_conceptizator_concepts.size()}")

    print("Model-specific sizes: ")
    # print(f"tensor_conceptizator_threshold: {tensor_conceptizator_threshold}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_images.pt"), tensor_to_save=tensor_images
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_concepts.pt"), tensor_to_save=tensor_concepts
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_preds.pt"), tensor_to_save=tensor_preds
    )

    utils.save_tensor(path=os.path.join(output_path, f"{mode}_tensor_y.pt"), tensor_to_save=tensor_y)
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_conceptizator_concepts.pt"),
        tensor_to_save=tensor_conceptizator_concepts
    )

    # utils.save_tensor(
    #     path=os.path.join(output_path, f"{mode}_tensor_conceptizator_threshold.pt"),
    #     tensor_to_save=tensor_conceptizator_threshold
    # )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_concept_mask.pt"),
        tensor_to_save=tensor_concept_mask
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_alpha.pt"),
        tensor_to_save=tensor_alpha
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_alpha_norm.pt"), tensor_to_save=tensor_alpha_norm
    )


def train_explainer(args):
    print("###############################################")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", root, "explainer")
    output_path = os.path.join(args.output, args.dataset, "Baseline", root, "explainer")
    tb_logs_path = os.path.join(args.logs, args.dataset, "Baseline", f"{root}_explainer")

    print("########### Paths ###########")
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    print("########### Paths ###########")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")

    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["train_transform"]
    val_transform = transforms["val_transform"]
    train_loader, val_loader = utils_dataset.get_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        train_transform,
        val_transform
    )

    if args.arch == "ResNet101":
        baseline = BaseLine_CUB(args).to(device)
        baseline.load_state_dict(torch.load(os.path.join(args.bb_chkpt, args.bb_chkpt_file)))
    else:
        baseline = VisionTransformer_baseline(
            CONFIGS[args.arch], args.img_size, zero_head=True, num_classes=len(args.labels),
            op_size=len(args.concept_names), smoothing_value=args.smoothing_value
        ).to(device)
        baseline.load_state_dict(torch.load(os.path.join(args.bb_chkpt, args.bb_chkpt_file))["model"])

    args.input_size_pi = 2048

    g = Gated_Logic_Net(
        args.input_size_pi,
        args.concept_names,
        args.labels,
        args.hidden_nodes,
        args.conceptizator,
        args.temperature_lens,
    ).to(device)

    solver = torch.optim.SGD(g.explainer.parameters(), lr=args.lr_explainer, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    logger = Logger_CUBS(
        1, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader, len(args.labels), device
    )

    fit_explainer(
        args.epochs_g,
        baseline,
        g,
        criterion,
        solver,
        train_loader,
        val_loader,
        logger,
        args.dataset,
        run_id,
        device
    )


def fit_explainer(
        epochs,
        baseline,
        g,
        criterion,
        solver,
        train_loader,
        val_loader,
        run_manager,
        dataset,
        run_id,
        device
):
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        running_loss = 0
        baseline.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (image, label, attributes) in enumerate(train_loader):
                image, label = image.to(device), label.to(torch.long).to(device)
                solver.zero_grad()

                logits_concepts = baseline(image)
                y_hat, _, _ = g(logits_concepts)
                train_loss = criterion(y_hat, label)
                train_loss.backward()
                solver.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(y_hat, label)

                running_loss += train_loss.item()
                t.set_postfix(epoch='{0}'.format(epoch), training_loss='{:05.3f}'.format(running_loss))
                t.update()

        baseline.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (image, label, attributes) in enumerate(val_loader):
                    image, label = image.to(device), label.to(torch.long).to(device)
                    logits_concepts = baseline(image)
                    y_hat, _, _ = g(logits_concepts)
                    val_loss = criterion(y_hat, label)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(y_hat, label)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(g)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)}")

    run_manager.end_run()


def train_backbone(args):
    print("###############################################")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", root, args.arch)
    output_path = os.path.join(args.output, args.dataset, "Baseline", root, args.arch)
    tb_logs_path = os.path.join(args.logs, args.dataset, "Baseline", f"{root}_{args.arch}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    device = utils.get_device()
    print(f"Device: {device}")

    transforms = utils.get_train_val_transforms(args.dataset, args.img_size, args.arch)
    train_transform = transforms["train_transform"]
    val_transform = transforms["val_transform"]
    train_loader, val_loader = utils_dataset.get_dataloader(
        args.data_root,
        args.json_root,
        args.dataset,
        args.bs,
        train_transform,
        val_transform
    )

    net = BaseLine_CUB(args).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    solver = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    schedule = utils.get_scheduler(solver, args)
    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_CUBS(1, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader, len(args.labels))
    run_manager.set_n_attributes(len(args.concept_names))
    fit(
        args.epochs,
        net,
        criterion,
        solver,
        schedule,
        train_loader,
        val_loader,
        run_manager,
        args.dataset,
        run_id,
        device
    )


def fit(
        epochs,
        net,
        criterion,
        solver,
        schedule,
        train_loader,
        val_loader,
        run_manager,
        dataset,
        run_id,
        device
):
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        running_loss = 0
        net.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                solver.zero_grad()
                image, attribute = utils.get_image_attributes(data_tuple, dataset)
                image = image.to(device, dtype=torch.float)
                attribute = attribute.to(device, dtype=torch.float)
                logits_concepts = net(image)
                train_loss = criterion(logits_concepts, attribute)
                train_loss.backward()
                solver.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_multilabel_per_epoch(torch.sigmoid(logits_concepts), attribute)

                running_loss += train_loss.item()
                t.set_postfix(epoch='{0}'.format(epoch), training_loss='{:05.3f}'.format(running_loss))
                t.update()

        net.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data_tuple in enumerate(val_loader):
                    image, attribute = utils.get_image_attributes(data_tuple, dataset)
                    image = image.to(device, dtype=torch.float)
                    attribute = attribute.to(device, dtype=torch.float)
                    logits_concepts = net(image)
                    val_loss = criterion(logits_concepts, attribute)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_multilabel_per_epoch(torch.sigmoid(logits_concepts),
                                                                             attribute)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        if schedule is not None:
            schedule.step()
        run_manager.end_epoch(net, multi_label=True)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} (%)  "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)}")

    run_manager.end_run()
