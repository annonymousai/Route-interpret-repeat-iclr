import os
from collections import defaultdict

import numpy as np
import torch

from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.residual import Residual


def get_glts(iteration, args, device, disease_folder, dataset="CUB"):
    glt_list = []
    for i in range(iteration - 1):
        # chk_pt_path = os.path.join(chk_pt_explainer, f"iter{i + 1}", "g", args.checkpoint_model[i])
        if dataset == "mimic_cxr":
            if i == 0:
                chk_pt_path = os.path.join(
                    args.checkpoints, args.dataset, "explainer", disease_folder, args.prev_chk_pt_explainer_folder[i],
                    f"iter{i + 1}", "g", "selected", args.metric, args.checkpoint_model[i]
                )
            elif i == 1:
                chk_pt_path = os.path.join(
                    args.checkpoints, args.dataset, "explainer", disease_folder,
                    args.prev_chk_pt_explainer_folder[i],
                    f"iter{i + 1}", "prev_cov_0.5", "g", "selected", args.metric, args.checkpoint_model[i]
                )

        else:
            chk_pt_path = os.path.join(
                args.checkpoints, args.dataset, "explainer", disease_folder, args.prev_chk_pt_explainer_folder[i],
                f"iter{i + 1}", "g", args.checkpoint_model[i]
            )
        print(f"===> glt for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        if dataset == "CUB":
            glt.load_state_dict(torch.load(chk_pt_path))
        elif dataset == "mimic_cxr":
            glt.load_state_dict(torch.load(chk_pt_path)["state_dict"])
        glt.eval()
        glt_list.append(glt)

    return glt_list


def get_residual(iteration, args, residual_chk_pt_path, device, dataset="CUB"):
    prev_residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
    if dataset == "mimic_cxr":
        residual_chk_pt = os.path.join(
            residual_chk_pt_path, "selected", args.metric, args.checkpoint_residual[-1]
        )
    else:
        residual_chk_pt = os.path.join(residual_chk_pt_path, args.checkpoint_residual[-1])
    print(f"=======> Residual loaded from: {residual_chk_pt}")
    # iteration - 2 = because we need to fetch the last residual, i.e (iteration -1)th residual and
    # the index of the array starts with 0. For example if iteration = 2, we need to fetch the 1st residual.
    # However, the residual_array index starts with 0, so we have to subtract 2 from current iteration.
    if dataset == "CUB" or dataset == "CIFAR10":
        prev_residual.load_state_dict(torch.load(residual_chk_pt))
    elif dataset == "mimic_cxr":
        prev_residual.load_state_dict(torch.load(residual_chk_pt)["state_dict"])

    prev_residual.eval()
    return prev_residual


def get_previous_pi_vals(iteration, glt_list, concepts):
    pi = []
    for i in range(iteration - 1):
        _, out_select, _ = glt_list[i](concepts)
        pi.append(out_select)

    return pi


def get_glts_for_all(iteration, args, device, g_chk_pt_path):
    glt_list = []
    for i in range(iteration - 1):
        chk_pt_path = os.path.join(g_chk_pt_path, f"iter{i + 1}", "explainer", args.checkpoint_model[i])
        print(f"===> glt for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        model_chk_pt = torch.load(chk_pt_path)
        if "state_dict" in model_chk_pt:
            glt.load_state_dict(model_chk_pt['state_dict'])
        else:
            glt.load_state_dict(model_chk_pt)
        glt.eval()
        glt_list.append(glt)

    return glt_list


def get_glts_for_HAM10k(iteration, args, device):
    glt_list = []
    for i in range(iteration - 1):
        # chk_pt_path = os.path.join(args.prev_explainer_chk_pt_folder[i], "explainer", args.checkpoint_model[i])
        if args.dataset == "HAM10k":
            chk_pt_path = os.path.join(
                args.prev_explainer_chk_pt_folder[i], "explainer", "accuracy", args.checkpoint_model[i]
            )
        else:
            chk_pt_path = os.path.join(
                args.prev_explainer_chk_pt_folder[i], "explainer", args.checkpoint_model[i]
            )
        print(f"=======> glt for iteration {i + 1} is loaded from {chk_pt_path}")
        glt = Gated_Logic_Net(
            args.input_size_pi,
            args.concept_names,
            args.labels,
            args.hidden_nodes,
            args.conceptizator,
            args.temperature_lens
        ).to(device)
        model_chk_pt = torch.load(chk_pt_path)
        if "state_dict" in model_chk_pt:
            glt.load_state_dict(model_chk_pt['state_dict'])
        else:
            glt.load_state_dict(model_chk_pt)
        glt.eval()
        glt_list.append(glt)

    return glt_list


# def get_residual_for_all(iteration, args, residual_chk_pt_path, device, dataset="CUB"):
#     prev_residual = Residual(args.dataset, args.pretrained, len(args.labels), args.arch).to(device)
#     residual_chk_pt = os.path.join(residual_chk_pt_path, args.checkpoint_residual[-1])
#     print(f"---> Residual loaded from: {residual_chk_pt}")
#     # iteration - 2 = because we need to fetch the last residual, i.e (iteration -1)th residual and
#     # the index of the array starts with 0. For example if iteration = 2, we need to fetch the 1st residual.
#     # However, the residual_array index starts with 0, so we have to subtract 2 from current iteration.
#     if dataset == "CUB":
#         prev_residual.load_state_dict(torch.load(residual_chk_pt))
#     elif dataset == "mimic_cxr":
#         prev_residual.load_state_dict(torch.load(residual_chk_pt)["state_dict"])
#
#     prev_residual.eval()
#     return prev_residual


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]
