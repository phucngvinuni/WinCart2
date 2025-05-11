# localization_algorithms.py
import math
import config
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid # Sử dụng tên gốc
from pathfinding.finder.a_star import AStarFinder

def rssi_distance_euclidean(rssi_vec1, rssi_vec2):
    """Tính khoảng cách Euclide giữa hai vector RSSI."""
    if len(rssi_vec1) != len(rssi_vec2):
        raise ValueError("Các vector RSSI phải có cùng độ dài")
    squared_diff_sum = sum([(v1 - v2)**2 for v1, v2 in zip(rssi_vec1, rssi_vec2)])
    return math.sqrt(squared_diff_sum)

def predict_location_knn(observed_rssi, fingerprints_data, k, weighted=False, epsilon=1e-6):
    """Dự đoán vị trí dựa trên KNN."""
    if not fingerprints_data:
        # print("Lỗi: Dữ liệu fingerprint trống.")
        return None

    distances_to_fingerprints = []
    for (r_fp, c_fp), rssi_fp_values in fingerprints_data.items():
        dist = rssi_distance_euclidean(observed_rssi, rssi_fp_values)
        distances_to_fingerprints.append(((r_fp, c_fp), dist))

    distances_to_fingerprints.sort(key=lambda item: item[1])
    k_nearest = distances_to_fingerprints[:k]

    if not k_nearest:
        # print("Lỗi: Không tìm thấy láng giềng nào.")
        return None

    if not weighted:
        sum_r, sum_c = 0, 0
        for (r_n, c_n), _ in k_nearest:
            sum_r += r_n
            sum_c += c_n
        estimated_r = sum_r / k
        estimated_c = sum_c / k
    else:
        weighted_sum_r, weighted_sum_c, sum_weights = 0, 0, 0
        for (r_n, c_n), dist_rssi in k_nearest:
            weight = 1 / (dist_rssi + epsilon)
            weighted_sum_r += r_n * weight
            weighted_sum_c += c_n * weight
            sum_weights += weight
        if sum_weights == 0:
            sum_r, sum_c = 0, 0
            for (r_n, c_n), _ in k_nearest:
                sum_r += r_n
                sum_c += c_n
            estimated_r = sum_r / k
            estimated_c = sum_c / k
            # print("Cảnh báo: Tổng trọng số bằng 0, sử dụng KNN không trọng số.")
        else:
            estimated_r = weighted_sum_r / sum_weights
            estimated_c = weighted_sum_c / sum_weights
    return (estimated_r, estimated_c)

def find_path_astar(grid_map_with_obstacles, start_node_grid, end_node_grid):
    """
    Tìm đường đi ngắn nhất bằng thuật toán A*.
    grid_map_with_obstacles: Bản đồ lưới của bạn (0 là lối đi, 1 là kệ).
    start_node_grid: (hàng, cột) của điểm bắt đầu.
    end_node_grid: (hàng, cột) của điểm kết thúc.
    """
    matrix = []
    for r_idx in range(grid_map_with_obstacles.shape[0]):
        row_data = []
        for c_idx in range(grid_map_with_obstacles.shape[1]):
            if grid_map_with_obstacles[r_idx, c_idx] == config.CELL_TYPE_PATH:
                row_data.append(1)
            else:
                row_data.append(0)
        matrix.append(row_data)

    path_grid_obj = Grid(matrix=matrix)

    start_pf_node_col, start_pf_node_row = start_node_grid[1], start_node_grid[0]
    end_pf_node_col, end_pf_node_row = end_node_grid[1], end_node_grid[0]

    try:
        start_node_obj = path_grid_obj.node(start_pf_node_col, start_pf_node_row)
        if not start_node_obj.walkable: # walkable được thư viện pathfinding đặt dựa trên giá trị matrix
            print(f"Lỗi tìm đường: Điểm bắt đầu ({start_node_grid[0]},{start_node_grid[1]}) là vật cản (theo pathfinding).")
            return None
    except IndexError:
        print(f"Lỗi tìm đường: Điểm bắt đầu ({start_node_grid[0]},{start_node_grid[1]}) nằm ngoài biên của pathfinding grid.")
        return None

    try:
        end_node_obj = path_grid_obj.node(end_pf_node_col, end_pf_node_row)
        if not end_node_obj.walkable:
            print(f"Lỗi tìm đường: Điểm kết thúc ({end_node_grid[0]},{end_node_grid[1]}) là vật cản (theo pathfinding).")
            return None
    except IndexError:
        print(f"Lỗi tìm đường: Điểm kết thúc ({end_node_grid[0]},{end_node_grid[1]}) nằm ngoài biên của pathfinding grid.")
        return None

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start_node_obj, end_node_obj, path_grid_obj)

    if path:
        return [(node.y, node.x) for node in path] # Chuyển lại (hàng, cột)
    return None