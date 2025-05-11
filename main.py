# main.py
import numpy as np
import time
import matplotlib.pyplot as plt
import config
import map_utils
import rssi_simulation
import localization_algorithms
import visualization

current_interactive_plot_obj = None
current_grid_map_data = None
current_access_points_list = None
current_rssi_fingerprints_map = None
current_item_locations_dict = None
current_map_num_rows = None
current_map_num_cols = None


def handle_map_click(actual_cart_pos_grid):
    global current_interactive_plot_obj, current_grid_map_data, current_access_points_list
    global current_rssi_fingerprints_map, current_item_locations_dict

    if current_interactive_plot_obj is None:
        return

    current_interactive_plot_obj.cart_actual_pos_grid = actual_cart_pos_grid
    cart_observed_rssi = rssi_simulation.get_observed_rssi_at_cart(
        actual_cart_pos_grid, current_grid_map_data, current_access_points_list
    )
    print(f"RSSI quan sát được (mới): {[round(val, 1) for val in cart_observed_rssi]}")

    estimated_pos_float = localization_algorithms.predict_location_knn(
        cart_observed_rssi,
        current_rssi_fingerprints_map,
        config.K_NEIGHBORS,
        config.USE_WEIGHTED_KNN,
        config.EPSILON_WEIGHT
    )

    if estimated_pos_float:
        current_interactive_plot_obj.cart_estimated_pos_float = estimated_pos_float
        error_m = rssi_simulation.euclidean_distance_m(actual_cart_pos_grid, estimated_pos_float)
        current_interactive_plot_obj.error_m = error_m
        print(f"Vị trí ước tính (ô): {estimated_pos_float}, Sai số: {error_m:.2f}m")
        current_interactive_plot_obj.update_plot_elements()

        item_names_available = list(current_item_locations_dict.keys())
        print("\nCác món hàng có sẵn:")
        for i, name in enumerate(item_names_available):
            if current_item_locations_dict[name]: # Chỉ hiển thị nếu có vị trí tiếp cận
                 print(f"{i+1}. {name} (Điểm tiếp cận: {current_item_locations_dict[name][0]})")
            else:
                 print(f"{i+1}. {name} (KHÔNG CÓ ĐIỂM TIẾP CẬN)")


        while True:
            try:
                choice = input(f"Nhập số TT món hàng bạn muốn tìm (hoặc 'q' để bỏ qua): ")
                if choice.lower() == 'q':
                    current_interactive_plot_obj.target_item_name = None
                    current_interactive_plot_obj.target_item_pos_grid = None
                    current_interactive_plot_obj.current_path_nodes = None
                    current_interactive_plot_obj.update_plot_elements()
                    break
                item_index = int(choice) - 1
                if 0 <= item_index < len(item_names_available):
                    selected_item_name = item_names_available[item_index]
                    # Kiểm tra xem món hàng đã chọn có vị trí tiếp cận không
                    if not current_item_locations_dict[selected_item_name]:
                        print(f"Món hàng '{selected_item_name}' không có điểm tiếp cận được định nghĩa. Vui lòng chọn món khác.")
                        current_interactive_plot_obj.update_plot_elements() # Xóa target cũ nếu có
                        continue # Quay lại vòng lặp hỏi


                    current_interactive_plot_obj.target_item_name = selected_item_name
                    print(f"Bạn đã chọn: {selected_item_name}")

                    target_pos = map_utils.get_item_target_location(
                        selected_item_name,
                        current_item_locations_dict,
                        estimated_pos_float # Truyền vị trí xe đẩy để chọn target gần nhất
                    )
                    if target_pos:
                        current_interactive_plot_obj.target_item_pos_grid = target_pos
                        print(f"Vị trí tiếp cận của '{selected_item_name}': {target_pos}")

                        start_node_for_path = (round(estimated_pos_float[0]), round(estimated_pos_float[1]))
                        if current_grid_map_data[start_node_for_path[0], start_node_for_path[1]] == config.CELL_TYPE_SHELF:
                            print(f"Cảnh báo: Điểm bắt đầu tìm đường {start_node_for_path} là kệ. Dùng vị trí thực tế.")
                            start_node_for_path = actual_cart_pos_grid

                        print(f"Tìm đường từ {start_node_for_path} đến {target_pos}...")
                        path_nodes = localization_algorithms.find_path_astar(
                            current_grid_map_data,
                            start_node_for_path,
                            target_pos
                        )
                        if path_nodes:
                            current_interactive_plot_obj.current_path_nodes = path_nodes
                            print(f"Đã tìm thấy đường đi gồm {len(path_nodes)} bước.")
                            current_interactive_plot_obj.update_plot_elements()
                            simulate_cart_movement(path_nodes, actual_cart_pos_grid)
                        else:
                            current_interactive_plot_obj.current_path_nodes = None
                            print(f"Không tìm thấy đường đi đến '{selected_item_name}' từ {start_node_for_path}.")
                            current_interactive_plot_obj.update_plot_elements()
                    else:
                        print(f"Không tìm thấy vị trí tiếp cận cho '{selected_item_name}'.")
                        current_interactive_plot_obj.target_item_pos_grid = None
                        current_interactive_plot_obj.current_path_nodes = None
                        current_interactive_plot_obj.update_plot_elements()
                    break
                else:
                    print("Lựa chọn không hợp lệ.")
            except ValueError:
                print("Vui lòng nhập một số hoặc 'q'.")
            except Exception as e:
                print(f"Đã xảy ra lỗi không mong muốn: {e}")
                break
    else:
        print("Không thể định vị xe đẩy sau khi click.")
        current_interactive_plot_obj.cart_estimated_pos_float = None
        current_interactive_plot_obj.error_m = None
        current_interactive_plot_obj.update_plot_elements()

def simulate_cart_movement(path_nodes, initial_actual_cart_pos):
    global current_interactive_plot_obj, current_grid_map_data, current_access_points_list, current_rssi_fingerprints_map

    if not path_nodes or current_interactive_plot_obj is None:
        return

    print("\nBắt đầu mô phỏng di chuyển xe đẩy...")
    current_interactive_plot_obj.current_path_nodes = path_nodes # Hiển thị toàn bộ đường đi

    for i, step_pos_grid in enumerate(path_nodes):
        current_interactive_plot_obj.cart_actual_pos_grid = step_pos_grid

        observed_rssi_at_step = rssi_simulation.get_observed_rssi_at_cart(
            step_pos_grid, current_grid_map_data, current_access_points_list
        )
        estimated_pos_at_step = localization_algorithms.predict_location_knn(
            observed_rssi_at_step, current_rssi_fingerprints_map,
            config.K_NEIGHBORS, config.USE_WEIGHTED_KNN, config.EPSILON_WEIGHT
        )

        if estimated_pos_at_step:
            current_interactive_plot_obj.cart_estimated_pos_float = estimated_pos_at_step
            error_m_at_step = rssi_simulation.euclidean_distance_m(step_pos_grid, estimated_pos_at_step)
            current_interactive_plot_obj.error_m = error_m_at_step
            # print(f"  Bước {i+1}/{len(path_nodes)}: Xe ở ({step_pos_grid[0]*config.GRID_RESOLUTION_M:.1f}, {step_pos_grid[1]*config.GRID_RESOLUTION_M:.1f}). "
            #       f"Ước tính: ({estimated_pos_at_step[0]*config.GRID_RESOLUTION_M:.1f}, {estimated_pos_at_step[1]*config.GRID_RESOLUTION_M:.1f}). Sai số: {error_m_at_step:.2f}m")

        current_interactive_plot_obj.update_plot_elements()
        current_interactive_plot_obj.fig.canvas.flush_events()
        time.sleep(0.2) # Giảm thời gian chờ để nhanh hơn

    print("Hoàn thành di chuyển đến món hàng.")
    # current_interactive_plot_obj.current_path_nodes = None # Tùy chọn: Xóa đường đi sau khi đến
    current_interactive_plot_obj.update_plot_elements()


def run_simulation():
    global current_interactive_plot_obj, current_grid_map_data, current_access_points_list
    global current_rssi_fingerprints_map, current_item_locations_dict, current_map_num_rows, current_map_num_cols

    current_grid_map_data, current_map_num_rows, current_map_num_cols = map_utils.create_base_map()

    # Định nghĩa thông tin kệ và các món hàng trên đó
    shelves_layout = [
        {
            'name': 'Kệ Trái Giữa', 'r': current_map_num_rows // 4, 'c': current_map_num_cols // 4,
            'rows': current_map_num_rows // 2, 'cols': 2, # Kệ dọc
            'items_on_shelf': [
                # Giả sử sữa và bánh mì dùng chung điểm tiếp cận của kệ này
                {'item_name': 'Sữa', 'preferred_side': 'right'},
                {'item_name': 'Bánh mì', 'preferred_side': 'right'}
            ]
        },
        {
            'name': 'Kệ Phải Giữa', 'r': current_map_num_rows // 4, 'c': (current_map_num_cols * 3 // 4) - 2, # Gần 3/4 cột
            'rows': current_map_num_rows // 2, 'cols': 2, # Kệ dọc
            'items_on_shelf': [
                {'item_name': 'Nước ngọt', 'preferred_side': 'left'}
            ]
        }
    ]

    for shelf in shelves_layout:
        current_grid_map_data = map_utils.add_shelf(
            current_grid_map_data, shelf['r'], shelf['c'], shelf['rows'], shelf['cols']
        )

    current_access_points_list = map_utils.define_access_points(current_map_num_rows, current_map_num_cols)
    current_item_locations_dict = map_utils.define_item_locations(
        current_grid_map_data, current_map_num_rows, current_map_num_cols, shelves_layout
    )

    print("Đang tạo bản đồ RSSI fingerprints...")
    current_rssi_fingerprints_map = rssi_simulation.generate_rssi_fingerprints(
        current_grid_map_data, current_access_points_list, current_map_num_rows, current_map_num_cols
    )
    print("Hoàn thành tạo bản đồ RSSI fingerprints.")

    print("\nBản đồ đã sẵn sàng. Click vào một ô lối đi để đặt xe đẩy.")
    print("Sau khi click, kiểm tra terminal để nhập món hàng cần tìm.")

    current_interactive_plot_obj = visualization.create_and_show_interactive_map(
        current_grid_map_data.copy(), # Truyền bản sao để tránh thay đổi không mong muốn
        current_access_points_list,
        current_item_locations_dict,
        current_rssi_fingerprints_map,
        current_map_num_rows,
        current_map_num_cols,
        handle_map_click
    )
    plt.show()

    print("Chương trình mô phỏng kết thúc.")

if __name__ == "__main__":
    run_simulation()    