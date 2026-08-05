import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomSegmentation(Dataset):
    """
    Custom dataset class for semantic segmentation
    """
    
    # กำหนดคลาสและสีตามที่คุณระบุ
    class_to_id = {
        "drivable Area": 0,
        "drivable area": 0,
        "traffic cone": 1,
        "car": 2,
        "person": 3,
        "slidewalk": 4,
        "parking": 5,
        "crosswalk": 6,
        "vegetation": 7,
        "golf cart": 8
    }
    
    id_to_color = {
    0: [0, 255, 0],      # green for drivable Area (RGB)
    1: [255, 0, 0],      # red for traffic cone (RGB)
    2: [128, 0, 128],    # purple for car (RGB)
    3: [0, 0, 255],      # blue for person (RGB)
    4: [192, 192, 192],  # gray for slidewalk (RGB)
    5: [0, 255, 255],    # cyan for parking (RGB)
    6: [255, 0, 255],    # magenta for crosswalk (RGB)
    7: [0, 128, 0],      # dark green for vegetation (RGB)
    8: [0, 0, 0]         # black for golf cart (RGB)
}
    
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (string): Directory with all the images and masks
            split (string): 'train' or 'val' split
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(self.root, f'{split}/images')
        self.masks_dir = os.path.join(self.root, f'{split}/masks')
        
        self.images = [os.path.join(self.images_dir, x) for x in os.listdir(self.images_dir) if x.endswith(('.png', '.jpg', '.jpeg'))]
        self.masks = [os.path.join(self.masks_dir, x) for x in os.listdir(self.masks_dir) if x.endswith(('.png', '.jpg', '.jpeg'))]
        
        # ตรวจสอบว่าจำนวนภาพและมาสก์เท่ากัน
        assert len(self.images) == len(self.masks), "Number of images and masks should be the same"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')  # convert to RGB if using color masks

        # 🔁 Resize mask if size mismatch
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        # Convert color mask to class label (H, W)
        mask_np = np.array(mask)
        label = np.zeros(mask_np.shape[:2], dtype=np.int64)

        for id, color in self.id_to_color.items():
            mask_match = np.all(mask_np == color, axis=2)
            label[mask_match] = id

        if self.transform:
            # 💡 ส่ง PIL.Image ทั้ง image และ label เข้า transform
            image_pil = Image.fromarray(np.array(image))
            label_pil = Image.fromarray(label.astype(np.uint8), mode='L')
            image, label = self.transform(image_pil, label_pil)

        return image, label

    @classmethod
    def decode_target(cls, target):
        """
        แปลง target ที่เป็น ID (0-8) กลับเป็นรูปสี BGR
        """
        target = target.astype(np.uint8)
        r = np.zeros_like(target, dtype=np.uint8)
        g = np.zeros_like(target, dtype=np.uint8)
        b = np.zeros_like(target, dtype=np.uint8)
        
        for id, color in cls.id_to_color.items():
            r[target == id] = color[0]
            g[target == id] = color[1]
            b[target == id] = color[2]
        
        rgb = np.stack([r, g, b], axis=2)
        return rgb
