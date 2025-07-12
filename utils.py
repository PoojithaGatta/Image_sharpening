import numpy as np

def tensor_to_img(tensor):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C, H, W -> H, W, C
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img
