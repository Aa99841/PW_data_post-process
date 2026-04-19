import cv2
import os
import numpy as np
from skan import Skeleton, summarize

def padding_Replication(img, size=230):
    
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

def pruning_skeleton_small(skeleton_cv):
    skeleton_bool = (skeleton_cv > 0)

    skel = Skeleton(skeleton_bool)
    stats = summarize(skel, separator='_') 

    pruned_skeleton = skeleton_cv.copy()
    height, width = skeleton_cv.shape

    for _ in range(3): 
        skel = Skeleton(pruned_skeleton > 0)
        stats = summarize(skel, separator='_')
        
        for i in range(len(stats)):
            branch_type = stats.loc[i, 'branch_type']
            branch_dist = stats.loc[i, 'branch_distance']
            
            if branch_type == 1: # 端點到分岔點
                coords = skel.path_coordinates(i)
                endpoint = coords[-1]
                
                # 邊界保護邏輯
                is_near_border = (endpoint[1] < 5) or (endpoint[1] > width - 5)
                
                # 如果不是邊界附近的短毛刺，就把它抹除
                if branch_dist < 30 and not is_near_border:
                    for r, c in coords.astype(int):
                        pruned_skeleton[r, c] = 0
    return pruned_skeleton

def pruning_skeleton(skeleton_cv):
    if not np.any(skeleton_cv > 0):
        return skeleton_cv

    try:
        skel = Skeleton(skeleton_cv > 0)
        stats = summarize(skel, separator='_') 
        if stats.empty:
            return skeleton_cv
        
        pruned_skeleton = skeleton_cv.copy()
        
        tips = stats[stats['branch_type'] == 1] # 1 代表 端點到節點 (分岔)

        if not tips.empty:
            tips_sorted = tips.sort_values(by='branch_distance', ascending=False)
            to_prune = tips_sorted.iloc[2:] # 將第三個頭（分岔）剪掉

            for idx in to_prune.index:
                coords = skel.path_coordinates(idx)
                for r, c in coords.astype(int):
                    pruned_skeleton[r, c] = 0 
                    
        return pruned_skeleton

    except Exception as e:
        print(f"剪枝失敗: {e}")
        return skeleton_cv

def find_absolute_angle_clipping(binary_mask, skeleton_img, position_ratio=0.5, absolute_angle_deg=45):
    """
    穿過血管中心線點，畫一條指定絕對角度的直線，並找到與邊界的交點。
    """
    if not np.any(skeleton_img > 0):
        return None

    skel = Skeleton(skeleton_img > 0)
    stats = summarize(skel, separator='_')
    if stats.empty: return None
    
    # 這裡加入防錯：有時候 SkeletonID 會變，確保抓到有資料的 ID
    if 'branch_distance' not in stats.columns: return None
    
    main_path_idx = stats['branch_distance'].idxmax()
    path_coords = skel.path_coordinates(main_path_idx)
    
    target_idx = int(len(path_coords) * position_ratio)
    # 確保索引不越界
    target_idx = max(0, min(len(path_coords)-1, target_idx))
    center_r, center_c = path_coords[target_idx]
    
    h, w = binary_mask.shape
    angle_rad = np.deg2rad(absolute_angle_deg)
    directions = [angle_rad, angle_rad + np.pi]
    intersection_pts = []

    for theta in directions:
        found_boundary = False
        # 增加搜尋範圍到對角線長度，確保一定能射出 Mask
        max_dist = int(np.sqrt(h**2 + w**2))
        for dist in np.arange(1, max_dist, 0.5):
            curr_r = int(center_r - dist * np.sin(theta)) 
            curr_c = int(center_c + dist * np.cos(theta))
            
            if not (0 <= curr_r < h and 0 <= curr_c < w):
                # 射出邊界前若沒碰到背景，紀錄邊界點用來畫線
                edge_r = max(0, min(h-1, curr_r))
                edge_c = max(0, min(w-1, curr_c))
                if not found_boundary: intersection_pts.append((edge_c, edge_r))
                break
                
            if binary_mask[curr_r, curr_c] == 0:
                intersection_pts.append((curr_c, curr_r))
                found_boundary = True
                break
        
    return (int(center_c), int(center_r)), intersection_pts

# ======= 設定路徑 =======

input_dir = r"C:\collega\Project\post_precessor\masks_cleaned"  
input_dir_ori = r"C:\collega\Project\post_precessor\images"
output_dir = "openCV/openCV_RangeGate_results_on_Original" 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ======= 遍歷資料夾 =======
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(f"找到 {len(image_files)} 張 Mask 圖片，開始對應原圖處理...")

for filename in image_files:
    # ======= 讀取圖像並二值化 ======= 
    mask_path = os.path.join(input_dir, filename)
    img_mask = cv2.imread(mask_path, 0) # 灰階讀取
    
    if img_mask is None:
        print(f"無法讀取 Mask: {filename}，跳過。")
        continue

    _, binary = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)

    # ======= 讀取原血管圖像 ======= 
    ori_filename = filename.replace("_label", "") 
    ori_path = os.path.join(input_dir_ori, ori_filename)
    img_ori = cv2.imread(ori_path) # 彩色讀取
    
    if img_ori is None:
        print(f"警告：找不到對應的原圖 {filename}，將在 Mask 上繪圖。")
        vis_img = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img_ori.copy()
        
    #  ======= padding ======= 
    size = 230
    bin_padR = padding_Replication(binary, size)
    
    #  ======= 使用 OpenCV 進行骨架化 =======
    skeleton_cv = cv2.ximgproc.thinning(bin_padR, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # ======= 裁切成原尺寸 =======
    h_m, w_m = img_mask.shape
    pad_h = (size - h_m) // 2
    pad_w = (size - w_m) // 2
    skeleton_cv = skeleton_cv[pad_h:pad_h+h_m , pad_w:pad_w+w_m]

    # ======= 剪枝 ======= 
    pruned_skeleton = pruning_skeleton(skeleton_cv)
    pruned_skeleton = pruning_skeleton_small(pruned_skeleton)

    # ======= 計算 Range Gate  =======
    target_ratio = 0.5  # 血管中點
    my_angle = -60  # 指定絕對角度

    result = find_absolute_angle_clipping(binary, pruned_skeleton, target_ratio, my_angle)

    # ======= 畫到原圖上 =======
    if result:
        center, inter_pts = result
        
        # 1. 畫穿過中心點的指定角度直線 (淺黃色，增加對比)
        if len(inter_pts) >= 2:
            cv2.line(vis_img, inter_pts[0], inter_pts[-1], (100, 255, 255), 1, cv2.LINE_AA)
            
        # 2. 標記中心點 (鮮綠色)
        cv2.circle(vis_img, center, 2, (0, 255, 0), -1)
        
        # 3. 標記與血管邊界的交點 (鮮紅色)
        for pt in inter_pts:
            if 0 <= pt[1] < binary.shape[0] and 0 <= pt[0] < binary.shape[1]:
                if binary[pt[1], pt[0]] == 0: # 碰到二值化邊界(黑色)才標記
                    cv2.circle(vis_img, pt, 2, (0, 0, 255), -1)

    # 3. 儲存結果
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, vis_img)
    
    print(f"已處理完成: {filename}")

print(f"--- 所有檔案處理完畢！結果儲存於 {output_dir} ---")
