import cv2
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
            # 這裡記得用你剛修正過的底線符號
            branch_type = stats.loc[i, 'branch_type']
            branch_dist = stats.loc[i, 'branch_distance']
            
            if branch_type == 1:  # 端點到分岔點
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

def find_absolute_angle_clipping(binary_mask, skeleton_img, position_ratio=0.5, absolute_angle_deg=45):
    """
    穿過血管中心線點，畫一條指定絕對角度的直線，並找到與邊界的交點。
    :param binary_mask: 原始二值化圖 (血管255, 背景0)
    :param skeleton_img: 處理後的中心線 (255)
    :param position_ratio: 中心線上的位置 (0.5 為中點)
    :param absolute_angle_deg: 這條線的絕對傾斜角度 (相對於圖片水平軸，度)
    """
    if not np.any(skeleton_img > 0):
        return None

    # 1. 提取中心線主幹並找到旋轉中心 (Center Point)
    skel = Skeleton(skeleton_img > 0)
    stats = summarize(skel, separator='_')
    if stats.empty: return None
    
    main_path_idx = stats['branch_distance'].idxmax()
    path_coords = skel.path_coordinates(main_path_idx) # [row, col]
    
    target_idx = int(len(path_coords) * position_ratio)
    center_r, center_c = path_coords[target_idx]
    
    # 2. 準備搜尋方向
    h, w = binary_mask.shape
    # 將角度轉為弧度 (Radians)
    angle_rad = np.deg2rad(absolute_angle_deg)
    
    # 定義兩個相反的方向 (Angle 和 Angle + 180)
    directions = [angle_rad, angle_rad + np.pi]
    intersection_pts = []

    # 3. 射線搜尋 (Ray Casting) 尋找交點
    for theta in directions:
        found_boundary = False
        # 從中心點向外延伸，步長 0.5 像素以提高精確度
        for dist in np.arange(1, max(h, w), 0.5):
            # 三角函數計算絕對座標 (注意 OpenCV 的 Y 軸向下，sin 要加負號)
            curr_r = int(center_r - dist * np.sin(theta)) 
            curr_c = int(center_c + dist * np.cos(theta))
            
            # 邊界檢查
            if not (0 <= curr_r < h and 0 <= curr_c < w):
                break
                
            # 交點偵測：從 255 (血管) 變成了 0 (背景)
            if binary_mask[curr_r, curr_c] == 0:
                intersection_pts.append((curr_c, curr_r)) # (x, y)
                found_boundary = True
                break
        
        # 如果射線射到圖片邊緣都沒碰到邊界，我們手動加上一個遠點用來畫線，但不標記為交點
        if not found_boundary:
             far_r = int(center_r - max(h, w) * np.sin(theta))
             far_c = int(center_c + max(h, w) * np.cos(theta))
             far_r = max(0, min(h-1, far_r))
             far_c = max(0, min(w-1, far_c))
             intersection_pts.append((far_c, far_r)) # 用來畫線的臨時點

    return (int(center_c), int(center_r)), intersection_pts

# ======= 執行與視覺化 =======
img = cv2.imread("masks_cleaned/data1_clip_423s_1.94s_label.png", 0)
img_ora = cv2.imread("images/data1_clip_423s_1.94s.png")
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#  ======= padding ======= 
bin_padR = padding_Replication(binary)

#  ======= 使用 OpenCV 進行骨架化 =======
skeleton_cv = cv2.ximgproc.thinning(bin_padR, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

# ======= 裁切成原尺寸 ======= 
cutx1, cutx2 = 3, 227
cuty1, cuty2 = 3, 227
skeleton_cv = skeleton_cv[cuty1:cuty2 , cutx1:cutx2]

# ======= 剪枝 ======= 
pruned_skeleton = pruning_skeleton(skeleton_cv)
pruned_skeleton = pruning_skeleton_small(pruned_skeleton)

# ======= 畫 range gate ======= 
target_ratio = 0.5     # 血管中點
my_angle = -60          # 指定這條穿過線的絕對角度為 60 度

result = find_absolute_angle_clipping(binary, pruned_skeleton, target_ratio, my_angle)

if result:
    center, inter_pts = result
    vis_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # --- 精簡繪圖：細線、小點 ---
    # 1. 畫穿過中心點的指定角度直線 
    if len(inter_pts) == 2:
        cv2.line(vis_img, inter_pts[0], inter_pts[1], (100, 150, 150), 1, cv2.LINE_AA)
        
    # 2. 標記中心點 (綠色, 半徑1)
    cv2.circle(vis_img, center, 2, (0, 255, 0), -1)
    
    # 3. 標記與血管邊界的交點 (紅色, 半徑2)
    for pt in inter_pts:
        if binary[pt[1], pt[0]] == 0: # 碰到邊界(黑色)才標記
            cv2.circle(vis_img, pt, 2, (0, 0, 255), -1)

    cv2.imshow("Specified Angle Clipping", vis_img)
    cv2.waitKey(0)
