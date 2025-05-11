# rssi_simulation.py
import math
import numpy as np
import config

def euclidean_distance_m(p1_grid, p2_grid):
    """Tính khoảng cách Euclide giữa hai điểm trên lưới (tính bằng mét)."""
    # p1_grid và p2_grid là tuple (hàng, cột)
    dist_pixels = math.sqrt((p1_grid[0] - p2_grid[0])**2 + (p1_grid[1] - p2_grid[1])**2)
    return dist_pixels * config.GRID_RESOLUTION_M

def get_line_cells(x1_col, y1_row, x2_col, y2_row): # Nhận (cột, hàng) cho Bresenham
    """Sử dụng thuật toán Bresenham để lấy danh sách các ô trên đường thẳng."""
    points = []
    # Chuyển đổi nội bộ sang (x,y) mà Bresenham thường dùng
    x1, y1 = x1_col, y1_row
    x2, y2 = x2_col, y2_row

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((y1, x1)) # Trả về (hàng, cột)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

def count_shelf_intersections(ap_pos_grid, cell_pos_grid, current_grid_map):
    """Đếm số lượng ô kệ hàng mà đường thẳng từ AP đến cell_pos_grid đi qua."""
    line_cells = get_line_cells(ap_pos_grid[1], ap_pos_grid[0], cell_pos_grid[1], cell_pos_grid[0])
    shelf_crossings = 0
    for r, c in line_cells[1:-1]:
        if 0 <= r < current_grid_map.shape[0] and 0 <= c < current_grid_map.shape[1]:
            if current_grid_map[r, c] == config.CELL_TYPE_SHELF:
                shelf_crossings += 1
    return shelf_crossings

def calculate_single_rssi(ap_pos_grid, cell_pos_grid, current_grid_map):
    """Tính toán RSSI mô phỏng tại cell_pos_grid từ một AP."""
    distance_m = euclidean_distance_m(ap_pos_grid, cell_pos_grid)

    if distance_m < config.GRID_RESOLUTION_M / 2: # Ở rất gần hoặc trùng AP
        return config.P_TX_MAX_RSSI + np.random.normal(0, config.NOISE_STD_DEV_DB / 3)

    path_loss_db = 10 * config.PATH_LOSS_EXPONENT_N * math.log10(distance_m)
    num_shelves = count_shelf_intersections(ap_pos_grid, cell_pos_grid, current_grid_map)
    total_shelf_attenuation_db = num_shelves * config.SHELF_ATTENUATION_DB
    noise_db = np.random.normal(0, config.NOISE_STD_DEV_DB)
    rssi = config.P_TX_MAX_RSSI - path_loss_db - total_shelf_attenuation_db + noise_db
    return max(rssi, config.MIN_RSSI_THRESHOLD)

def generate_rssi_fingerprints(grid_map, access_points, num_rows, num_cols):
    """Tạo bản đồ fingerprint RSSI cho tất cả các ô lối đi."""
    fingerprints = {}
    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            if grid_map[r_idx, c_idx] != config.CELL_TYPE_SHELF:
                current_cell_rssi_values = []
                for ap_pos in access_points:
                    rssi_val = calculate_single_rssi(ap_pos, (r_idx, c_idx), grid_map)
                    current_cell_rssi_values.append(rssi_val)
                fingerprints[(r_idx, c_idx)] = current_cell_rssi_values
    return fingerprints

def get_observed_rssi_at_cart(cart_pos_grid, grid_map, access_points):
    """Tính toán RSSI 'quan sát được' tại vị trí xe đẩy."""
    observed_rssi = []
    for ap_pos in access_points:
        rssi_val = calculate_single_rssi(ap_pos, cart_pos_grid, grid_map)
        observed_rssi.append(rssi_val)
    return observed_rssi