import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_train_image(image, mask, title=None, axes=True):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    ax[0].imshow(image) 
    ax[1].imshow(image)
    ax[1].imshow(mask[..., 0], alpha=0.7, cmap='gray')
    if axes:
        for a in ax:
             a.set_xticks([]); a.set_yticks([])
    if title is not None:
        plt.suptitle(title, fontsize=16)
    plt.show();


def plot_losses(losses):
    _, axes = plt.subplots(ncols=2, figsize=(15, 6))

    for i, key in enumerate(losses.keys()):
            axes[0].plot([v['loss'] for v in losses[key]], label='{} loss'.format(key), alpha=0.7, color='C{}'.format(i))    

    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid()

    for i, key in enumerate(losses.keys()):
        names = list(losses[key][0].keys())
        names.remove('loss')
        for name in names:
            axes[1].plot([v[name] for v in losses[key]], label='{} {}'.format(key, name), alpha=0.7, color='C{}'.format(i))

    axes[1].set_title('Meterics')
    axes[1].legend()
    axes[1].grid()

    plt.show()


# Functions to visualize bounding boxes and class labels on an image. 
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize_bboxes(pred, annotations, category_id_to_name):
    _, axes = plt.subplots(ncols=2, figsize=(15, 7))
    img = annotations['img'][..., 0]
    if not isinstance(img, np.ndarray):
        img = annotations['img'].data.numpy().copy()[0]
    img = np.dstack([img] * 3)
    img1, img2 = img.copy(), img.copy()
    for idx, bbox in enumerate(pred['annot']):
        img1 = visualize_bbox(img1, bbox, 1, category_id_to_name)
    for idx, bbox in enumerate(annotations['annot']):
        img2 = visualize_bbox(img2, bbox, 1, category_id_to_name)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    plt.show()


def plot_retina_losses_w_clf(losses):
    _, axes = plt.subplots(ncols=3, figsize=(21, 6))
    axes[0].plot([l['clf_loss'] for l in losses['train_losses']], label='clf loss train', alpha=0.8)
    axes[0].plot([l['meters']['clf_loss'] for l in losses['val_losses']], label='clf loss val', alpha=0.8)

    axes[0].set_title('Clf Loss')
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot([l['loss'] for l in losses['train_losses']], label='weighted loss train', alpha=0.8)
    axes[1].plot([l['bbx_reg_loss'] for l in losses['train_losses']], label='reg loss train', alpha=0.5)
    axes[1].plot([l['bbx_clf_loss'] for l in losses['train_losses']], label='clf loss train', alpha=0.5)

    axes[1].plot([l['meters']['loss'] for l in losses['val_losses']], label='weighted loss val', alpha=0.8)
    axes[1].plot([l['meters']['bbx_reg_loss'] for l in losses['val_losses']], label='reg loss val', alpha=0.5)
    axes[1].plot([l['meters']['bbx_clf_loss'] for l in losses['val_losses']], label='clf loss val', alpha=0.5)

    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid()

    axes[2].plot([l['iou'] for l in losses['val_losses']], label='mAP', alpha=0.7)

    axes[2].set_title('Meterics')
    axes[2].legend()
    axes[2].grid()

    plt.show()

def plot_retina_losses(losses):
    _, axes = plt.subplots(ncols=2, figsize=(21, 6))
    axes[0].plot([l['loss'] for l in losses['train_losses']], label='weighted loss train', alpha=0.8)
    axes[0].plot([l['bbx_reg_loss'] for l in losses['train_losses']], label='reg loss train', alpha=0.5)
    axes[0].plot([l['bbx_clf_loss'] for l in losses['train_losses']], label='clf loss train', alpha=0.5)

    axes[0].plot([l['meters']['loss'] for l in losses['val_losses']], label='weighted loss val', alpha=0.8)
    axes[0].plot([l['meters']['bbx_reg_loss'] for l in losses['val_losses']], label='reg loss val', alpha=0.5)
    axes[0].plot([l['meters']['bbx_clf_loss'] for l in losses['val_losses']], label='clf loss val', alpha=0.5)

    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot([l['iou'] for l in losses['val_losses']], label='mAP', alpha=0.7)

    axes[1].set_title('Meterics')
    axes[1].legend()
    axes[1].grid()

    plt.show()
