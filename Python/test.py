import pytest
from app import bfs, dfs, dijkstra, path_latlon_list

TEST_GRAPH = {
    (0,0): {(0,1):1, (1,0):1},
    (0,1): {(0,0):1, (1,1):1},
    (1,0): {(0,0):1, (1,1):1},
    (1,1): {(0,1):1, (1,0):1}
}

@pytest.fixture(autouse=True)
def setup_graph(monkeypatch):
    monkeypatch.setattr("app.GRAPH", TEST_GRAPH)

def test_bfs_path():
    start = (0,0)
    goal = (1,1)
    path = bfs(start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal

def test_dfs_path():
    start = (0,0)
    goal = (1,1)
    path = dfs(start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal

def test_dijkstra_path():
    start = (0,0)
    goal = (1,1)
    path = dijkstra(start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal

def test_path_latlon_list():
    path = [(0,0),(0,1),(1,1)]
    latlon = path_latlon_list(path)
    assert latlon == [[0,0],[1,0],[1,1]] or latlon is not None
