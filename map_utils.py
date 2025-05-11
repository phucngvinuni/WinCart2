# map_utils.py
import numpy as np
import config

def create_base_map():
    """Tạo bản đồ lưới cơ sở với kích thước đã định nghĩa."""
    num_cols = int(config.SUPERMARKET_WIDTH_M / config.GRID_RESOLUTION_M)
    num_rows = int(config.SUPERMARKET_HEIGHT_M / config.GRID_RESOLUTION_M)
    grid_map = np.full((num_rows, num_cols), config.CELL_TYPE_PATH, dtype=int) # Mặc định là lối đi
    return grid_map, num_rows, num_cols

def add_shelf(grid_map, row_start, col_start, num_shelf_rows, num_shelf_cols):
    """Thêm một kệ hàng hình chữ nhật vào bản đồ."""
    max_row, max_col = grid_map.shape
    end_row = min(row_start + num_shelf_rows, max_row)
    end_col = min(col_start + num_shelf_cols, max_col)
    grid_map[row_start : end_row, col_start : end_col] = config.CELL_TYPE_SHELF
    return grid_map

def define_access_points(num_rows, num_cols):
    """Định nghĩa vị trí của 4 AP ở 4 góc."""
    margin = config.AP_MARGIN_CELLS
    access_points = [
        (margin, margin),
        (margin, num_cols - 1 - margin),
        (num_rows - 1 - margin, margin),
        (num_rows - 1 - margin, num_cols - 1 - margin)
    ]
    return access_points

def find_accessible_spot_near_shelf(grid_map, shelf_r, shelf_c, shelf_num_rows, shelf_num_cols, preferred_side=None):
    """
    Tìm một ô lối đi gần nhất với một kệ hàng cụ thể.
    shelf_r, shelf_c: Tọa độ góc trên trái của kệ.
    shelf_num_rows, shelf_num_cols: Kích thước của kệ.
    preferred_side: 'left', 'right', 'top', 'bottom'
    """
    num_rows_map, num_cols_map = grid_map.shape
    candidate_spots = []

    shelf_cells = []
    for r_offset in range(shelf_num_rows):
        for c_offset in range(shelf_num_cols):
            r, c = shelf_r + r_offset, shelf_c + c_offset
            if 0 <= r < num_rows_map and 0 <= c < num_cols_map:
                 if grid_map[r,c] == config.CELL_TYPE_SHELF:
                    shelf_cells.append((r,c))

    if not shelf_cells:
        # print(f"Cảnh báo: Không có ô kệ nào được tìm thấy tại ({shelf_r},{shelf_c}) với kích thước ({shelf_num_rows},{shelf_num_cols}) để tìm điểm tiếp cận.")
        return None

    for r_sh, c_sh in shelf_cells:
        potential_neighbors = [
            (-1, 0, 'top'), (1, 0, 'bottom'),
            (0, -1, 'left'), (0, 1, 'right')
        ]
        for dr, dc, side_name in potential_neighbors:
            check_r, check_c = r_sh + dr, c_sh + dc
            if 0 <= check_r < num_rows_map and 0 <= check_c < num_cols_map and \
               grid_map[check_r, check_c] == config.CELL_TYPE_PATH:
                candidate_spots.append(((check_r, check_c), side_name))

    if not candidate_spots:
        # print(f"Cảnh báo: Không tìm thấy ô lối đi trực tiếp gần kệ ({shelf_r},{shelf_c}).")
        return None

    if preferred_side:
        sides_to_check = [preferred_side] if isinstance(preferred_side, str) else preferred_side
        for p_side in sides_to_check:
            preferred_spots_info = [spot_info for spot_info in candidate_spots if spot_info[1] == p_side]
            if preferred_spots_info:
                preferred_actual_spots = [spot for spot, side in preferred_spots_info]
                avg_r = sum(r for r,c in preferred_actual_spots) / len(preferred_actual_spots)
                avg_c = sum(c for r,c in preferred_actual_spots) / len(preferred_actual_spots)
                best_spot = min(preferred_actual_spots, key=lambda spot: \
                                     ((spot[0]-avg_r)**2 + (spot[1]-avg_c)**2) )
                return best_spot

    shelf_center_r = shelf_r + shelf_num_rows / 2
    shelf_center_c = shelf_c + shelf_num_cols / 2
    all_candidate_actual_spots = [spot for spot, side in candidate_spots]
    best_overall_spot = min(all_candidate_actual_spots, key=lambda spot: \
                             ((spot[0]-shelf_center_r)**2 + (spot[1]-shelf_center_c)**2) )
    return best_overall_spot


def define_item_locations(grid_map, num_rows, num_cols, shelves_layout_info): # Đổi tên tham số
    """
    Định nghĩa vị trí (ô lối đi có thể tiếp cận) cho các món hàng.
    shelves_layout_info: list of dicts, từ main.py
    """
    items_approachable_locations = {}

    for shelf_info in shelves_layout_info:
        shelf_r, shelf_c = shelf_info['r'], shelf_info['c']
        shelf_num_r, shelf_num_c = shelf_info['rows'], shelf_info['cols']

        for item_detail in shelf_info['items_on_shelf']:
            item_name = item_detail['item_name']
            preferred_side = item_detail.get('preferred_side')

            if item_name not in items_approachable_locations:
                items_approachable_locations[item_name] = []

            # Tìm điểm tiếp cận cho toàn bộ kệ này dựa trên preferred_side
            accessible_spot = find_accessible_spot_near_shelf(
                grid_map, shelf_r, shelf_c, shelf_num_r, shelf_num_c, preferred_side
            )

            if accessible_spot:
                if accessible_spot not in items_approachable_locations[item_name]:
                    items_approachable_locations[item_name].append(accessible_spot)
            else:
                 print(f"CẢNH BÁO: Không tìm được điểm tiếp cận cho kệ '{shelf_info.get('name', 'Không tên')}' chứa '{item_name}' với preferred_side='{preferred_side}'.")

    return items_approachable_locations


def get_item_target_location(item_name, item_locations_dict, current_cart_pos_grid=None):
    """
    Lấy một vị trí (ô lưới) cho món hàng được yêu cầu từ danh sách các điểm tiếp cận.
    """
    if item_name in item_locations_dict and item_locations_dict[item_name]:
        possible_targets = item_locations_dict[item_name]
        if not possible_targets:
            # print(f"Cảnh báo: '{item_name}' có trong từ điển nhưng danh sách vị trí trống.")
            return None

        if current_cart_pos_grid and len(possible_targets) > 1:
            best_target = min(possible_targets, key=lambda target_pos:
                              ((target_pos[0] - current_cart_pos_grid[0])**2 +
                               (target_pos[1] - current_cart_pos_grid[1])**2) )
            return best_target
        return possible_targets[0]
    # print(f"Cảnh báo: Không có vị trí tiếp cận nào được định nghĩa cho '{item_name}'.")
    return None