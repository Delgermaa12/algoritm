import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
import math


class UBPathFinder:
    def __init__(self):
        self.graph = None
        self.area_name = "Ulaanbaatar, Mongolia"

    def download_map_data(self):
        """Улаанбаатар хотын замын сүлжээг татаж авч байрлуулах"""
        print("Улаанбаатар хотын замын сүлжээг татаж авч байна...")

        try:
            # Улаанбаатар хотын замын сүлжээг татах
            self.graph = ox.graph_from_place(self.area_name, network_type='drive')
            print("✅ Замын сүлжээ амжилттай татагдлаа!")

            # Графикийг энгийн болгох
            self.graph = ox.utils_graph.get_undirected(self.graph)

            # Замын уртыг тооцоолох
            self.graph = ox.add_edge_lengths(self.graph)

            print(f"📊 График мэдээлэл: {len(self.graph.nodes)} орой, {len(self.graph.edges)} ирмэг")

        except Exception as e:
            print(f"❌ Алдаа: {e}")
            print("🔧 Интернэт холболтоо шалгана уу")

    def find_nodes_near_location(self, lat: float, lon: float) -> Optional[int]:
        """Өгөгдсөн координаттай ойролцоо оройг олох"""
        if self.graph is None:
            print("❌ График байхгүй байна")
            return None

        try:
            # Координатад хамгийн ойрхон оройг олох
            node_id = ox.distance.nearest_nodes(self.graph, lon, lat)
            print(f"📍 Координат ({lat}, {lon}) -> Орой {node_id}")
            return node_id
        except Exception as e:
            print(f"❌ Орой олоход алдаа: {e}")
            return None

    def bfs_shortest_path(self, start_node: int, end_node: int) -> Tuple[List[int], float, Dict]:
        """BFS алгоритм ашиглан хамгийн богино замыг олох"""
        print("🔄 BFS алгоритмаар замыг хайж байна...")
        start_time = time.time()
        stats = {'visited_nodes': 0, 'iterations': 0}

        try:
            # BFS ашиглан замыг олох
            path = nx.shortest_path(self.graph, start_node, end_node, method='dijkstra')
            stats['visited_nodes'] = len(path)

            # Замын уртыг тооцоолох
            path_length = self.calculate_path_length(path)

            end_time = time.time()
            execution_time = end_time - start_time
            stats['execution_time'] = execution_time

            print(f"✅ BFS: {execution_time:.4f} секунд, {len(path)} орой, {path_length:.1f} метр")

            return path, path_length, stats

        except nx.NetworkXNoPath:
            print("❌ BFS: Зам олдсонгүй")
            return [], float('inf'), stats
        except Exception as e:
            print(f"❌ BFS алдаа: {e}")
            return [], float('inf'), stats

    def dfs_path(self, start_node: int, end_node: int) -> Tuple[List[int], float, Dict]:
        """DFS алгоритм ашиглан замыг олох"""
        print("🔄 DFS алгоритмаар замыг хайж байна...")
        start_time = time.time()
        stats = {'visited_nodes': 0, 'iterations': 0}

        visited = set()
        stack = [(start_node, [start_node])]
        stats['iterations'] = 0

        while stack:
            stats['iterations'] += 1
            current_node, path = stack.pop()

            if current_node == end_node:
                # Замын уртыг тооцоолох
                path_length = self.calculate_path_length(path)
                stats['visited_nodes'] = len(visited)

                end_time = time.time()
                execution_time = end_time - start_time
                stats['execution_time'] = execution_time

                print(f"✅ DFS: {execution_time:.4f} секунд, {len(path)} орой, {path_length:.1f} метр")
                return path, path_length, stats

            if current_node not in visited:
                visited.add(current_node)

                # Хөрш оройнуудыг нэмэх
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))

        stats['visited_nodes'] = len(visited)
        print("❌ DFS: Зам олдсонгүй")
        return [], float('inf'), stats

    def dijkstra_shortest_path(self, start_node: int, end_node: int) -> Tuple[List[int], float, Dict]:
        """Dijkstra алгоритм ашиглан хамгийн богино замыг олох"""
        print("🔄 Dijkstra алгоритмаар замыг хайж байна...")
        start_time = time.time()
        stats = {'visited_nodes': 0, 'iterations': 0}

        try:
            # Dijkstra ашиглан хамгийн богино замыг олох
            path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            stats['visited_nodes'] = len(path)

            # Замын уртыг тооцоолох
            path_length = self.calculate_path_length(path)

            end_time = time.time()
            execution_time = end_time - start_time
            stats['execution_time'] = execution_time

            print(f"✅ Dijkstra: {execution_time:.4f} секунд, {len(path)} орой, {path_length:.1f} метр")

            return path, path_length, stats

        except nx.NetworkXNoPath:
            print("❌ Dijkstra: Зам олдсонгүй")
            return [], float('inf'), stats
        except Exception as e:
            print(f"❌ Dijkstra алдаа: {e}")
            return [], float('inf'), stats

    def calculate_path_length(self, path: List[int]) -> float:
        """Замын нийт уртыг тооцоолох"""
        if len(path) < 2:
            return 0

        total_length = 0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                # Эхний ирмэгийн уртыг авах
                first_edge = next(iter(edge_data.values()))
                length = first_edge.get('length', 0)
                total_length += length

        return total_length

    def validate_path(self, path: List[int], start_node: int, end_node: int) -> bool:
        """Олдсон замын зөв эсэхийг шалгах"""
        if not path:
            return False

        # Эхлэл ба төгсгөл шалгах
        if path[0] != start_node or path[-1] != end_node:
            return False

        # Зам дахь бүх ирмэгүүд шалгах
        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return False

        return True

    def visualize_paths(self, start_node: int, end_node: int,
                        bfs_path: List[int], dfs_path: List[int],
                        dijkstra_path: List[int], bfs_stats: Dict,
                        dfs_stats: Dict, dijkstra_stats: Dict):
        """Гурван алгоритмын үр дүнг харьцуулан харуулах"""
        if self.graph is None:
            print("❌ График байхгүй байна")
            return

        # 4x4 хүснэгт үүсгэх
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Улаанбаатар хотын замын сүлжээ - Алгоритмын харьцуулалт', fontsize=16, fontweight='bold')

        # Анхны замын сүлжээ
        ox.plot_graph(self.graph, ax=axes[0, 0], node_size=0, edge_color='gray',
                      edge_linewidth=0.3, show=False, close=False)
        axes[0, 0].set_title('🗺️ Замын сүлжээний бүтэц', fontsize=12, fontweight='bold')
        axes[0, 0].text(0.02, 0.98, f'Орой: {len(self.graph.nodes)}\nИрмэг: {len(self.graph.edges)}',
                        transform=axes[0, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # BFS замыг харуулах
        if bfs_path and self.validate_path(bfs_path, start_node, end_node):
            ox.plot_graph_route(self.graph, bfs_path, ax=axes[0, 1], route_color='red',
                                route_linewidth=4, node_size=0, show=False, close=False)
            bfs_length = self.calculate_path_length(bfs_path)
            axes[0, 1].set_title(f'🔴 BFS Алгоритм\n{bfs_length:.1f} метр, {bfs_stats["execution_time"]:.3f} сек',
                                 fontsize=12, fontweight='bold')
            axes[0, 1].text(0.02, 0.98,
                            f'Орой: {len(bfs_path)}\nЗай: {bfs_length:.0f}м\nЦаг: {bfs_stats["execution_time"]:.3f}с',
                            transform=axes[0, 1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        else:
            axes[0, 1].set_title('❌ BFS Алгоритм - Зам олдсонгүй', fontsize=12)

        # DFS замыг харуулах
        if dfs_path and self.validate_path(dfs_path, start_node, end_node):
            ox.plot_graph_route(self.graph, dfs_path, ax=axes[1, 0], route_color='blue',
                                route_linewidth=4, node_size=0, show=False, close=False)
            dfs_length = self.calculate_path_length(dfs_path)
            axes[1, 0].set_title(f'🔵 DFS Алгоритм\n{dfs_length:.1f} метр, {dfs_stats["execution_time"]:.3f} сек',
                                 fontsize=12, fontweight='bold')
            axes[1, 0].text(0.02, 0.98,
                            f'Орой: {len(dfs_path)}\nЗай: {dfs_length:.0f}м\nЦаг: {dfs_stats["execution_time"]:.3f}с',
                            transform=axes[1, 0].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            axes[1, 0].set_title('❌ DFS Алгоритм - Зам олдсонгүй', fontsize=12)

        # Dijkstra замыг харуулах
        if dijkstra_path and self.validate_path(dijkstra_path, start_node, end_node):
            ox.plot_graph_route(self.graph, dijkstra_path, ax=axes[1, 1], route_color='green',
                                route_linewidth=4, node_size=0, show=False, close=False)
            dijkstra_length = self.calculate_path_length(dijkstra_path)
            axes[1, 1].set_title(
                f'🟢 Dijkstra Алгоритм\n{dijkstra_length:.1f} метр, {dijkstra_stats["execution_time"]:.3f} сек',
                fontsize=12, fontweight='bold')
            axes[1, 1].text(0.02, 0.98,
                            f'Орой: {len(dijkstra_path)}\nЗай: {dijkstra_length:.0f}м\nЦаг: {dijkstra_stats["execution_time"]:.3f}с',
                            transform=axes[1, 1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            axes[1, 1].set_title('❌ Dijkstra Алгоритм - Зам олдсонгүй', fontsize=12)

        plt.tight_layout()
        plt.show()

    def print_comparison_table(self, bfs_stats: Dict, dfs_stats: Dict, dijkstra_stats: Dict,
                               bfs_length: float, dfs_length: float, dijkstra_length: float):
        """Харьцуулалтын хүснэгт хэвлэх"""
        print("\n" + "=" * 80)
        print("📊 АЛГОРИТМЫН ХАРЬЦУУЛАЛТЫН ХҮСНЭГТ")
        print("=" * 80)
        print(f"{'Алгоритм':<12} {'Замны урт (м)':<15} {'Гүйцэтгэл (сек)':<15} {'Давуу тал':<30}")
        print("-" * 80)

        # BFS мэдээлэл
        bfs_advantage = "• Бүх боломжит зам • Гаранттай шийдэл"
        print(f"{'BFS':<12} {bfs_length:<15.1f} {bfs_stats['execution_time']:<15.4f} {bfs_advantage:<30}")

        # DFS мэдээлэл
        dfs_advantage = "• Гүнзгий хайлт • Санах ой бага"
        print(f"{'DFS':<12} {dfs_length:<15.1f} {dfs_stats['execution_time']:<15.4f} {dfs_advantage:<30}")

        # Dijkstra мэдээлэл
        dijkstra_advantage = "• Хамгийн богино зам • Жинтэй график"
        print(
            f"{'Dijkstra':<12} {dijkstra_length:<15.1f} {dijkstra_stats['execution_time']:<15.4f} {dijkstra_advantage:<30}")

        print("-" * 80)

        # Хамгийн богино замыг тодруулах
        min_length = min(bfs_length, dfs_length, dijkstra_length)
        if min_length != float('inf'):
            if dijkstra_length == min_length:
                print("🎉 Dijkstra алгоритм хамгийн богино замыг оллоо!")
            elif bfs_length == min_length:
                print("🎉 BFS алгоритм хамгийн богино замыг оллоо!")
            else:
                print("🎉 DFS алгоритм хамгийн богино замыг оллоо!")

        # Хамгийн хурдан алгоритмыг тодруулах
        times = [bfs_stats['execution_time'], dfs_stats['execution_time'], dijkstra_stats['execution_time']]
        min_time = min(times)
        algorithms = ['BFS', 'DFS', 'Dijkstra']
        fastest_algo = algorithms[times.index(min_time)]
        print(f"⚡ {fastest_algo} алгоритм хамгийн хурдан ажиллалаа: {min_time:.4f} секунд")

    def test_algorithm_correctness(self, start_node: int, end_node: int):
        """Алгоритмуудын зөв эсэхийг шалгах тест"""
        print("\n" + "🔍 АЛГОРИТМЫН ЗӨВ БАЙДЛЫН ШАЛГАЛТ")
        print("-" * 50)

        # Бүх алгоритмаар замыг олох
        bfs_path, bfs_length, _ = self.bfs_shortest_path(start_node, end_node)
        dfs_path, dfs_length, _ = self.dfs_path(start_node, end_node)
        dijkstra_path, dijkstra_length, _ = self.dijkstra_shortest_path(start_node, end_node)

        # Шалгуурууд
        tests_passed = 0
        total_tests = 0

        # Тест 1: Зам эхлэх цэгээс эхэлсэн эсэх
        total_tests += 1
        if bfs_path and bfs_path[0] == start_node:
            tests_passed += 1
            print("✅ BFS: Зам зөв эхлэх цэгээс эхэлж байна")
        else:
            print("❌ BFS: Зам буруу эхлэх цэгтэй")

        total_tests += 1
        if dfs_path and dfs_path[0] == start_node:
            tests_passed += 1
            print("✅ DFS: Зам зөв эхлэх цэгээс эхэлж байна")
        else:
            print("❌ DFS: Зам буруу эхлэх цэгтэй")

        total_tests += 1
        if dijkstra_path and dijkstra_path[0] == start_node:
            tests_passed += 1
            print("✅ Dijkstra: Зам зөв эхлэх цэгээс эхэлж байна")
        else:
            print("❌ Dijkstra: Зам буруу эхлэх цэгтэй")

        # Тест 2: Зам дуусах цэгт төгссөн эсэх
        total_tests += 1
        if bfs_path and bfs_path[-1] == end_node:
            tests_passed += 1
            print("✅ BFS: Зам зөв дуусах цэгт төгссөн")
        else:
            print("❌ BFS: Зам буруу дуусах цэгтэй")

        total_tests += 1
        if dfs_path and dfs_path[-1] == end_node:
            tests_passed += 1
            print("✅ DFS: Зам зөв дуусах цэгт төгссөн")
        else:
            print("❌ DFS: Зам буруу дуусах цэгтэй")

        total_tests += 1
        if dijkstra_path and dijkstra_path[-1] == end_node:
            tests_passed += 1
            print("✅ Dijkstra: Зам зөв дуусах цэгт төгссөн")
        else:
            print("❌ Dijkstra: Зам буруу дуусах цэгтэй")

        # Тест 3: Зам дахь бүх ирмэгүүд графанд байгаа эсэх
        total_tests += 1
        if self.validate_path(bfs_path, start_node, end_node):
            tests_passed += 1
            print("✅ BFS: Зам дахь бүх ирмэгүүд зөв")
        else:
            print("❌ BFS: Зам дахь зарим ирмэг буруу")

        total_tests += 1
        if self.validate_path(dfs_path, start_node, end_node):
            tests_passed += 1
            print("✅ DFS: Зам дахь бүх ирмэгүүд зөв")
        else:
            print("❌ DFS: Зам дахь зарим ирмэг буруу")

        total_tests += 1
        if self.validate_path(dijkstra_path, start_node, end_node):
            tests_passed += 1
            print("✅ Dijkstra: Зам дахь бүх ирмэгүүд зөв")
        else:
            print("❌ Dijkstra: Зам дахь зарим ирмэг буруу")

        print(f"\n📈 Шалгуурын үр дүн: {tests_passed}/{total_tests} амжилттай")

        return tests_passed == total_tests

    def compare_algorithms(self, start_lat: float, start_lon: float,
                           end_lat: float, end_lon: float):
        """Гурван алгоритмыг харьцуулах"""
        print("🚀 Улаанбаатар хотын замын сүлжээнд алгоритмуудыг харьцуулж байна...")

        # Эхлэх ба дуусах цэгүүдийн ойролцоох оройг олох
        start_node = self.find_nodes_near_location(start_lat, start_lon)
        end_node = self.find_nodes_near_location(end_lat, end_lon)

        if start_node is None or end_node is None:
            print("❌ Эхлэх эсвэл дуусах цэг олдсонгүй")
            return

        print(f"📍 Эхлэх цэг: {start_node}, Дуусах цэг: {end_node}")

        # Алгоритмуудын зөв эсэхийг шалгах
        correctness_test = self.test_algorithm_correctness(start_node, end_node)

        if not correctness_test:
            print("⚠️  Алгоритмуудын зөв байдлын шалгаралт амжилтгүй болсон тул үр дүнг харуулахгүй")
            return

        # Гурван алгоритмаар замыг олох
        bfs_path, bfs_length, bfs_stats = self.bfs_shortest_path(start_node, end_node)
        dfs_path, dfs_length, dfs_stats = self.dfs_path(start_node, end_node)
        dijkstra_path, dijkstra_length, dijkstra_stats = self.dijkstra_shortest_path(start_node, end_node)

        # Үр дүнг харьцуулах хүснэгт
        self.print_comparison_table(bfs_stats, dfs_stats, dijkstra_stats,
                                    bfs_length, dfs_length, dijkstra_length)

        # Дүрслэх
        self.visualize_paths(start_node, end_node, bfs_path, dfs_path,
                             dijkstra_path, bfs_stats, dfs_stats, dijkstra_stats)


def main():
    """Үндсэн програм"""
    print("=" * 60)
    print("🏙️  Улаанбаатар хотын замын сүлжээний алгоритмын харьцуулалт")
    print("=" * 60)

    path_finder = UBPathFinder()

    # Замын сүлжээг татах
    path_finder.download_map_data()

    if path_finder.graph is None:
        print("❌ Замын сүлжээг татаж авахад алдаа гарлаа. Дахин оролдоно уу.")
        return

    # Жишээ координатууд (Улаанбаатар хотын өөр өөр цэгүүд)
    print("\n🎯 Жишээ замын чиглэл:")

    # Сонголт 1: Төв цэгүүд
    print("1. Сүхбаатарын талбай -> Чингис хаан олон улсын нисэх онгоцны буудал")
    start_lat, start_lon = 47.9185, 106.9177  # Сүхбаатарын талбай
    end_lat, end_lon = 47.6467, 106.8197  # Нисэх онгоцны буудал

    # Сонголт 2: Хан-Уул дүүрэг -> Бага тойруу
    # start_lat, start_lon = 47.8900, 106.8900
    # end_lat, end_lon = 47.9300, 106.9300

    print(f"📍 Эхлэх: ({start_lat}, {start_lon})")
    print(f"🎯 Дуусах: ({end_lat}, {end_lon})")

    # Алгоритмуудыг харьцуулах
    path_finder.compare_algorithms(start_lat, start_lon, end_lat, end_lon)

    print("\n" + "=" * 60)
    print("✅ Програм амжилттай дууслаа!")
    print("=" * 60)


if __name__ == "__main__":
    main()