import cv2
import os
import numpy as np
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

def padding_Replication(img, size):    
    
    shape = img.shape
    h = shape[0]
    w = shape[1]
    
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    padded = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE,
        value=0
    )
    
    return padded

def pruning_skeleton(skeleton_cv):
    if not np.any(skeleton_cv > 0):
        return skeleton_cv

    try:
        # 1. 將二值化影像轉化為一個圖數據結構，識別端點、分岔點
        skel = Skeleton(skeleton_cv > 0)
        
        # 2. 列出圖中每一條「分支」的資訊
        stats = summarize(skel, separator='_') 
        if stats.empty:
            return skeleton_cv
        
        pruned_skeleton = skeleton_cv.copy()
        
        junctions = stats[stats['branch_type'] == 2] # 2 代表 節點到節點 (通常主幹中間)
        tips = stats[stats['branch_type'] == 1]      # 1 代表 端點到節點 (分岔)

        # 找到較短分支剪掉
        if not tips.empty:
            tips_sorted = tips.sort_values(by='branch_distance', ascending=False) # 將所有端點分支按長度由長到短排序
            
            to_prune = tips_sorted.iloc[2:] # 將第三個頭（分岔）剪掉

            for idx in to_prune.index:
                coords = skel.path_coordinates(idx) # 取得該分支所有像素的座標
                for r, c in coords.astype(int):
                    pruned_skeleton[r, c] = 0 
                    
        return pruned_skeleton

    except Exception as e:
        print(f"剪枝失敗: {e}")
        return skeleton_cv

# ======= 設定路徑 =======
input_dir = "masks_cleaned"   # 你的輸入資料夾
output_dir = "results_pruned" # 處理後的儲存位置

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ======= 開始遍歷資料夾 =======
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(f"找到 {len(image_files)} 張圖片，開始處理...")

for filename in image_files:
    # ======= 讀取圖像並二值化 ======= 
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, 0) # 以灰階讀取
    
    if img is None:
        continue
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    #  ======= padding ======= 
    size = 230
    bin_padR = padding_Replication(binary, size)

    #  ======= 使用skimage进行骨架提取 ======= 
    skeleton = skeletonize(bin_padR)
    skeleton_uint8 = (skeleton * 255).astype (np. uint8) # 轉乘unit8

    # ======= 裁切成原尺寸 =======  
    pad = int((size - 224) / 2)
    cutx1, cutx2 = pad, pad+224
    cuty1, cuty2 = pad, pad+224
    cropedR = skeleton_uint8[cuty1:cuty2 , cutx1:cutx2]

    # ======= 剪枝 ======= 
    pruned_skeleton = pruning_skeleton(cropedR)

    # ======= 畫回原圖 ======= 
    img_copy = img.copy()
    final_result = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    rows, cols = np.where(pruned_skeleton > 0)
    final_result[rows, cols] = [0,0,225]

    # 3. 儲存結果
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, final_result)
    
    print(f"已處理完成: {filename}")

print("--- 所有檔案處理完畢！ ---")

# def pruning_skeleton(skeleton_cv):
#     skeleton_bool = (skeleton_cv > 0)

#     skel = Skeleton(skeleton_bool)
#     stats = summarize(skel, separator='_')  

#     pruned_skeleton = skeleton_cv.copy()
#     height, width = skeleton_cv.shape

#     for _ in range(5): 
#         if not np.any(pruned_skeleton > 0):
#             print(f"第 {i} 次迭代：影像已無剩餘骨架像素，停止剪枝。")
#             break
        
#         skel = Skeleton(pruned_skeleton > 0)
#         stats = summarize(skel, separator='_')
        
#         for i in range(len(stats)):
#             # 這裡記得用你剛修正過的底線符號
#             branch_type = stats.loc[i, 'branch_type']
#             branch_dist = stats.loc[i, 'branch_distance']
            
#             if branch_type == 1:  # 端點到分岔點
#                 coords = skel.path_coordinates(i)
#                 endpoint = coords[-1]
                
#                 # 邊界保護邏輯
#                 is_near_border = (endpoint[1] < 5) or (endpoint[1] > width - 5)
                
#                 # 如果不是邊界附近的短毛刺，就把它抹除
#                 if branch_dist < 30 and not is_near_border:
#                     for r, c in coords.astype(int):
#                         pruned_skeleton[r, c] = 0
#     return pruned_skeleton
