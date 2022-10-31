import cv2
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt


visualise = True

class Node():
    def __init__(self, x, y, cost=0):
        self.x = x
        self.y = y
        self.cost = cost

class PathPlanner():
    def __init__(self, config):
        self.config = config
        self.x_max = self.config["x_max"]
        self.y_max = self.config["y_max"]
        self.start = Node(*self.config["start"])
        self.end = Node(*self.config["end"])
        self.static = []
        self.dynamic = []
        self.mem = []
        self.stay_lane = [
            Node(0, 1, 1), # move up
            Node(0, -1, 1), # move down
            Node(0, 0, 1.0) # stay
        ]
        self.change_lane = [
            Node(1, 3, 1.1), # move up and right
            Node(-1, 3, 1.1), # move up and left
            Node(1, 2, 1.1), # move up and right
            Node(-1, 2, 1.1), # move up and left
        ]

        self.dfs_search = [
            Node(1, 1, 1),
            Node(-1, 1, 1),
            Node(0, 1, 1),
        ]

        self.motions = [self.stay_lane, self.change_lane]

        self.base_map = np.zeros((self.config["y_max"], self.config["x_max"]))
        self.world = self.create_world()

    def create_world(self):
        world = np.copy(self.base_map)
        world = self.add_obstacles(world)
        world = self.add_dynamic_obstacles(world)
        return world

    def add_obstacles(self, world):
        # add a line of obstacles
        for i in range(90):
            # obstacle_map = np.copy(self.base_map)
            world[i, 0] = self.heuristic(Node(0, i), self.end) * 10
            self.static.append(Node(0, i))

        if visualise:
            plt.figure(figsize=(6, 100))
            for i in range(90):
                plt.scatter(0, i, c="black")
        return world

    def add_dynamic_obstacles(self, world):
        world[20, 1] = self.heuristic(Node(1, 20), self.end) * 10
        world[65, 1] = self.heuristic(Node(1, 65), self.end) * 10
        world[20, 2] = self.heuristic(Node(2, 20), self.end) * 10

        self.dynamic.append(Node(1, 20))
        self.dynamic.append(Node(1, 65))
        self.dynamic.append(Node(2, 20))

        if visualise:
            self.dynamic_obj_points = plt.scatter([1, 1, 2], [20, 65, 20], c="red")

        return world

    def update_world(self, changes):
        for obj in self.dynamic:
            self.world[obj.y][obj.x] = 0.0
        for i in range(len(changes)):
            self.dynamic[i] = Node(
                self.dynamic[i].x + changes[i].x, self.dynamic[i].y + changes[i].y
            )
            self.world[self.dynamic[i].y][self.dynamic[i].x] = math.inf

        if visualise:
            plot_x = [obj.x for obj in self.dynamic]
            plot_y = [obj.y for obj in self.dynamic]
            self.dynamic_obj_points.remove()
            self.dynamic_obj_points = plt.scatter(plot_x, plot_y, c="red")

    def is_obstacle(self, node):
        return any([self.compare_coordinates(node, obstacle)
                    for obstacle in self.static]) or \
               any([self.compare_coordinates(node, obstacle)
                    for obstacle in self.dynamic])

    def is_valid_point(self, node):
        if 0 <= node.x < self.x_max and 0 <= node.y < self.y_max:
            return True
        return False

    def is_same_direction(self, signs):
        if abs(sum(np.sign(signs))) == 2 and [sign == 0 for sign in signs]:
            return True
        else:
            return False

    def is_zigzagging(self, node, paths, path_len):
        node1 = paths[path_len - 2]
        node2 = paths[path_len - 1]
        if path_len > 1 and node1.x == node.x and not self.is_same_direction([node2.x - node1.x, node.x - node2.x]):
            return True
        else:
            return False

    def compare_coordinates(self, node1, node2):
        return node1.x == node2.x and node1.y == node2.y

    def add_coordinates(self, node1, node2):
        return Node(node1.x + node2.x, node1.y + node2.y)

    def get_neighbours(self, node):
        neighbours = []
        for motion in self.stay_lane:
            for increment in [0, 1, 2]:
                action = Node(motion.x, motion.y + increment, cost=motion.cost)
                n = self.add_coordinates(node, action)
                if not self.is_valid_point(n):
                    print(f"New node ({n.x}, {n.y}) to move to is not a valid point")
                    break
                if self.is_obstacle(n):
                    print(f"New node ({n.x}, {n.y}) to move to is an obstacle")
                    break
                n.cost = action.cost
                neighbours.append(n)

        for motion in self.change_lane:
            n = self.add_coordinates(node, motion)
            if not self.is_valid_point(n):
                    continue
            n.cost = motion.cost
            neighbours.append(n)
        return neighbours

    def get_neighbours_dfs(self, node):
        neighbours = []
        for action in self.dfs_search:
            n = self.add_coordinates(node, action)
            if not self.is_valid_point(n):
                continue
            n.cost = action.cost
            neighbours.append(n)
        return neighbours

    def heuristic(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def search_neighbours(self, neighbours):
        result = min(
            neighbours,
            key=lambda n:
            self.heuristic(n, self.goal) + n.cost + self.world[n.y][n.x]
        )
        return result
        
    def compute(self, current, goal):
        previous = None
        path = [current]
        self.goal = goal
        while not self.compare_coordinates(current, self.goal):
            # search neighbouring cells for shortest path
            neighbours = self.get_neighbours(current)
            result = self.search_neighbours(neighbours)
            # print(f"Result: {[result.x, result.y]}")
            previous = current
            # print(f"Previous: {[previous.x, previous.y]}")
            current = result
            # print(f"Current: {[current.x, current.y]}")
            path_len = len(path)
            # print(path_len)
            path.append(current)

        return path


    def dfs(self, path, current, goal):
        visited = path[:-1]
        previous = current
        ignored = []
        new_path = []
        while not self.compare_coordinates(current, goal):
            neighbours = self.get_neighbours_dfs(current)
            for n in neighbours:
                if (n.x, n.y) in [(v.x, v.y) for v in visited]:
                    neighbours.remove(n)
                if (n.x, n.y) in [(s.x, s.y) for s in self.static]:
                    neighbours.remove(n)
                if (n.x, n.y) in [(d.x, d.y) for d in self.dynamic]:
                    neighbours.remove(n)
            if len(neighbours) == 0:
                visited.append(current)
                current = previous
                continue
            next = self.search_neighbours(neighbours)
            for nn in neighbours:
                if self.compare_coordinates(nn, next):
                    neighbours.remove(nn)
            current = next
            ignored += neighbours
            new_path.append(current)
        return new_path
            



def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    print("Initialising path planner")
    pp = PathPlanner(config)

    # set the start and end points for the agent
    current = config["start"]
    print(f"Starting Point: {current}")
    goal = config["end"]
    print(f"Goal to reach: {goal}")
    
    # while the agent has not reached the end
    while current != goal:
        print("Planning path...")
        start = time.time()

        # update the positions of dynamic objects in the map
        pp.update_world([Node(0, 2), Node(0, 2), Node(0, 2)])
        
        # compute / recompute the shortest path to the end based on the agent's current location
        path = pp.compute(Node(*current), Node(*goal))

        # compute the next shortest path
        new_path = pp.dfs(path, Node(*current), Node(*goal))
        
        end = time.time()
        print(f"Compute time: {end - start}")

        if visualise:
            # Locate the agent
            agent_point = plt.scatter([current[0]], [current[1]], linewidth=4, color='red')
            
            # plot shortest path
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            path_points = plt.scatter(path_x, path_y, s=3, color="blue")

            # plot alternate path
            new_path_x = [p.x for p in new_path]
            new_path_y = [p.y for p in new_path]
            new_path_points = plt.scatter(new_path_x, new_path_y, s=3, color="green")

            plt.pause(0.001)
            # plt.waitforbuttonpress() # uncomment if want to see progression
            path_points.remove()
            agent_point.remove()
            new_path_points.remove()

        # move the agent to the next point in the path
        current = [path[1].x, path[1].y]

    # plot last position of agent
    agent_point = plt.scatter([current[0]], [current[1]], linewidth=4, color='red')
    print("Completed")
    plt.waitforbuttonpress()
    



if __name__ == "__main__":    
    main()
    

