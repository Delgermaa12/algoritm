# app.py
from flask import Flask, render_template, request, jsonify
import geopandas as gpd
from shapely.geometry import LineString, box
import heapq
from collections import deque
import os
import threading
from scipy.spatial import cKDTree

app = Flask(__name__)

SHAPE_PATH = "data/gis_osm_roads_free_1.shp"
UB_BBOX = {"minx": 106.75, "miny": 47.75, "maxx": 107.05, "maxy": 47.98}
PROJECTED_CRS = "EPSG:32647"   # for metric lengths
GEOGRAPHIC_CRS = "EPSG:4326"

GRAPH = None
NODE_LIST = None
KD_TREE = None
NODE_INDEX = None
GRAPH_LOCK = threading.Lock()

def build_graph_once():
    global GRAPH, NODE_LIST, KD_TREE, NODE_INDEX
    with GRAPH_LOCK:
        if GRAPH is not None:
            return

        if not os.path.exists(SHAPE_PATH):
            raise FileNotFoundError(f"{SHAPE_PATH} not found. Place roads.shp in data/")

        print("Loading shapefile...")
        gdf = gpd.read_file(SHAPE_PATH)
        gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

        if gdf.crs is None:
            gdf.set_crs(GEOGRAPHIC_CRS, inplace=True)
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(GEOGRAPHIC_CRS)

        bbox_geom = box(UB_BBOX["minx"], UB_BBOX["miny"], UB_BBOX["maxx"], UB_BBOX["maxy"])
        gdf = gdf[gdf.intersects(bbox_geom)]
        if len(gdf) == 0:
            raise RuntimeError("No road features found inside UB bounding box. Check SHAPE_PATH and UB_BBOX.")

        print(f"Features after clipping: {len(gdf)}")
        gdf_proj = gdf.to_crs(PROJECTED_CRS)

        adj = {}
        def add_edge(a,b,w):
            if a == b: return
            if b not in adj.setdefault(a, {}) or adj[a][b] > w:
                adj[a][b] = w
                adj.setdefault(b, {})
                adj[b][a] = w

        for geom_orig, geom_proj in zip(gdf.geometry, gdf_proj.geometry):
            if geom_orig is None or geom_proj is None: continue
            parts_orig = [geom_orig] if geom_orig.geom_type == "LineString" else list(geom_orig)
            parts_proj = [geom_proj] if geom_proj.geom_type == "LineString" else list(geom_proj)
            for p_orig, p_proj in zip(parts_orig, parts_proj):
                coords_orig = list(p_orig.coords)
                coords_proj = list(p_proj.coords)
                for i in range(len(coords_orig)-1):
                    a_lonlat = (coords_orig[i][0], coords_orig[i][1])
                    b_lonlat = (coords_orig[i+1][0], coords_orig[i+1][1])
                    seg = LineString([coords_proj[i], coords_proj[i+1]])
                    w = seg.length  # meters (projected CRS)
                    add_edge(a_lonlat, b_lonlat, w)

        nodes = list(adj.keys())
        if len(nodes) == 0:
            raise RuntimeError("Graph has 0 nodes after building - check shapefile and bbox.")
        coords = [(n[0], n[1]) for n in nodes]  # lon,lat

        tree = cKDTree(coords)
        index_map = { nodes[i]: i for i in range(len(nodes)) }

        GRAPH = adj
        NODE_LIST = nodes
        KD_TREE = tree
        NODE_INDEX = index_map

        print(f"Graph built: nodes={len(nodes)}")

def nearest_node(point):
    global NODE_LIST, KD_TREE
    if KD_TREE is None:
        build_graph_once()
    dist, idx = KD_TREE.query([point[0], point[1]], k=1)
    return NODE_LIST[int(idx)]

def bfs(start, goal):
    adj = GRAPH
    q = deque([start])
    parent = {start: None}
    while q:
        v = q.popleft()
        if v == goal:
            break
        for nb in adj.get(v, {}):
            if nb not in parent:
                parent[nb] = v
                q.append(nb)
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur); cur = parent[cur]
    path.reverse()
    return path

def dfs(start, goal):
    adj = GRAPH
    stack = [start]
    parent = {start: None}
    visited = set([start])
    while stack:
        v = stack.pop()
        if v == goal:
            break
        for nb in adj.get(v, {}):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = v
                stack.append(nb)
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur); cur = parent[cur]
    path.reverse()
    return path

def dijkstra(start, goal):
    adj = GRAPH
    pq = [(0, start)]
    dist = {start: 0}
    parent = {start: None}
    while pq:
        d, v = heapq.heappop(pq)
        if v == goal:
            break
        if d > dist.get(v, float('inf')): continue
        for nb, w in adj.get(v, {}).items():
            nd = d + w
            if nd < dist.get(nb, float('inf')):
                dist[nb] = nd
                parent[nb] = v
                heapq.heappush(pq, (nd, nb))
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur); cur = parent.get(cur)
    path.reverse()
    return path

def path_latlon_list(path):
    return [[p[1], p[0]] for p in path] if path else None

@app.route("/")
def index():
    threading.Thread(target=build_graph_once, daemon=True).start()
    return render_template("index.html")

@app.route("/find_json", methods=["POST"])
def find_json():
    data = request.get_json()
    start_lon = float(data["start_lon"])
    start_lat = float(data["start_lat"])
    goal_lon = float(data["goal_lon"])
    goal_lat = float(data["goal_lat"])
    alg = data.get("algorithm", "all")

    build_graph_once()

    start_node = nearest_node((start_lon, start_lat))
    goal_node = nearest_node((goal_lon, goal_lat))

    results = {}
    if alg in ("bfs", "all"):
        p = bfs(start_node, goal_node)
        results["bfs"] = path_latlon_list(p)
    if alg in ("dfs", "all"):
        p = dfs(start_node, goal_node)
        results["dfs"] = path_latlon_list(p)
    if alg in ("dijkstra", "all"):
        p = dijkstra(start_node, goal_node)
        results["dijkstra"] = path_latlon_list(p)

    return jsonify({"start":[start_lat, start_lon], "goal":[goal_lat, goal_lon], "paths": results})

if __name__ == "__main__":
    print("Starting app, building graph (may take some seconds)...")
    build_graph_once()
    app.run(debug=True)
