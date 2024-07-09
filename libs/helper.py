import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from libs.class_id_map import get_id2class_map
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from libs.postprocess import PostProcessor
from tqdm import tqdm
from prompt.tools import (segment_video_labels, gen_label, gen_label_split,
                          generate_segment_features,generate_split_features,
                          create_logits, split_feature, split_gt, split_gt_feature, split_mixed_class)
from prompt.text_prompt import text_prompt_for_clip

def train(
    train_loader: DataLoader,
    model: nn.Module,
    model_text: nn.Module,
    class_text_list,
    joint_text_list,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    criterion_contrast: nn.Module,
    lambda_bound_loss: float,
    optimizer: optim.Optimizer,
    dataset_name,
    device, output_device,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    # switch training mode
    model.train()

    for sample in tqdm(train_loader):
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]


        x = x.to(output_device)
        t = t.to(output_device)
        b = b.to(output_device)
        mask = mask.to(output_device)
        joint_text_list = joint_text_list.to(output_device)

        optimizer.zero_grad()

        batch_size = x.shape[0]

        joint_text_embedding = model_text(joint_text_list).float()

        # compute output and loss
        output_cls, output_bound, output_feature, output_feature_split, logit_scale = model(x, mask, joint_text_embedding)

        #Action-text pairs
        t_segment = segment_video_labels(t)

        label =  [i[0] for seg in t_segment for i in seg]

        label_g = gen_label(label)

        texts = list()
        for single_label in label:
            text_item = class_text_list[single_label].unsqueeze(dim=0)
            texts.append(text_item)

        texts = torch.cat(texts).cuda(output_device)
        text_embedding = model_text(texts).float()

        action_embeddings = []
        if isinstance(output_feature, list):
            for i in range(len(output_feature)):
                action_embedding = generate_segment_features(output_feature[i], t_segment, output_device)
                action_embeddings.append(action_embedding)

        #Clip-text pairs
        gt_split, feature_split = split_mixed_class(t_segment,2)

        feature_split_embedding = generate_split_features(output_feature_split, feature_split, output_device)

        text_split = text_prompt_for_clip(gt_split, dataset_name, "simple").cuda(output_device)

        text_split_embedding = model_text(text_split).float()

        label_split_g = gen_label_split(gt_split)



        loss = 0.0
        # Action segmentation loss
        if isinstance(output_cls, list):
            n = len(output_cls)
            for out in output_cls:
                loss += criterion_cls(out, t, x) / n
        else:
            loss += criterion_cls(output_cls, t, x)

        # boundary regression loss
        if isinstance(output_bound, list):
            n = len(output_bound)
            for out in output_bound:
                loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
        else:
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

        # action-text contrastive loss
        if isinstance(action_embeddings, list):
            for i in range(len(action_embeddings)):
                logits_per_image, logits_per_text = create_logits(action_embeddings[i], text_embedding, logit_scale[0])
                ground_truth = torch.tensor(label_g, dtype=action_embedding.dtype, device=output_device)

                loss_imgs = criterion_contrast(logits_per_image, ground_truth)
                loss_texts = criterion_contrast(logits_per_text, ground_truth)

                loss += 0.8 * ((loss_imgs + loss_texts) / 2)

        # clip-text contrastive loss
        logits_per_image, logits_per_text = create_logits(feature_split_embedding, text_split_embedding,
                                                          logit_scale[1])
        ground_truth = torch.tensor(label_split_g, dtype=feature_split_embedding.dtype, device=output_device)

        loss_imgs = criterion_contrast(logits_per_image, ground_truth)
        loss_texts = criterion_contrast(logits_per_text, ground_truth)

        loss += 0.5 * ((loss_imgs + loss_texts) / 2)

        # record loss
        losses.update(loss.item(), batch_size)


        loss.backward()
        optimizer.step()

    return losses.avg


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    model_text: nn.Module,
    joint_text_list,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    device,output_device,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    boundary_th: float,
    tolerance: int,
    refinement_method: Optional[str] = None
) -> Tuple[float, float, float, float, float, float, float, float, str]:
    losses = AverageMeter("Loss", ":.4e")
    postprocessor = PostProcessor(refinement_method, boundary_th)
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(output_device)
            t = t.to(output_device)
            b = b.to(output_device)
            mask = mask.to(output_device)
            joint_text_list = joint_text_list.to(output_device)

            batch_size = x.shape[0]

            joint_text_embedding = model_text(joint_text_list).float()

            # compute output and loss
            output_cls, output_bound = model(x, mask, joint_text_embedding)

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            ) #加上了边界的预测
            # update score
            scores_cls.update(output_cls, t, output_bound, mask) #The result of not utilizing boundary branch
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t) #The result of utilizing boundary branch

    cls_acc, edit_score, segment_f1s = scores_after_refinement.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s,
        bound_acc,
        precision,
        recall,
        bound_f1s,
    )

def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
    model_text,
    joint_text_list,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    config : str,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)

    scores_before_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)
            joint_text_list = joint_text_list.to(device)

            joint_text_embedding = model_text(joint_text_list).float()

            # compute output and loss
            output_cls, output_bound = model(x, mask, joint_text_embedding)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)
            
    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())

    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )
    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))
    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )