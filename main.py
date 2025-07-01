import copy
import keras
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

# Завантажуємо дані MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормалізація даних
x_train = x_train / 255.0
x_test = x_test / 255.00

y_train_cat = keras.utils.to_categorical(y_train,10)
y_test_cat = keras.utils.to_categorical(y_test,10)

# Додаємо канал (для ч/б зображень)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Створюємо модель згорткової нейронної мережі
model = keras.Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

])

# Компільована модель
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train_cat, epochs=5, validation_split=0.2)



class Graph():
    def __init__(self, nodes):
        self.nodes = nodes
        self.size = int(nodes**0.5)
        self.graph = nx.Graph()
        self.pos = {}
        self.edge_list = []
        self.speeds = {}
        self.speeds_history = {}
        # створюємо сітку
        for i in range(nodes):
            self.graph.add_node(f"{i}")
        c = 0
        for k,r in [(k,r) for k in range(self.size) for r in range(self.size)]:
            self.pos[f"{c}"] = (k, r)
            c += 1
        # з'єднуємо сітку
        for i in range(nodes):
            if (f"{i + 1}" in self.pos and ((i + 1) % self.size != 0)):
                self.graph.add_edge(f"{i}", f"{i + 1}")
                self.edge_list.append((f"{i}", f"{i + 1}"))
            if f"{i + self.size}" in self.pos:
                self.graph.add_edge(f"{i}", f"{i + self.size}")
                self.edge_list.append((f"{i}", f"{i + self.size}"))

        self.node_colors = ['lightblue' for _ in self.graph.nodes()]

    def removeEdge(self, number_nodes):
        random_edge_list = copy.deepcopy(self.edge_list)
        random.shuffle(random_edge_list)

        count_rem = 0
        while(count_rem < number_nodes):
            if (count_rem < self.nodes):
                edge = random_edge_list[count_rem]
                self.graph.remove_edge(edge[0], edge[1])
                if not nx.is_connected(self.graph):
                    self.graph.add_edge(edge[0], edge[1])
                    number_nodes += 1
                count_rem += 1
            else:
                break

    def assign_speeds(self):
        edges = list(gr.graph.edges())
        for edge in edges:
            number = random.randint(2, 9)
            indices = np.where(y_test == number)[0]
            if len(indices) == 0:
                print("Число не знайдено в вибірці.")
            else:
                image = x_test[indices[0]]

            self.speeds[edge] = image
            self.speeds_history[edge] = number

    def drawGraph(self, show_speeds=False):
        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw(self.graph, node_color=self.node_colors, pos=self.pos,
                node_size=1000, with_labels=True, ax=ax)

        if show_speeds:
            for edge, image in self.speeds.items():
                # Вираховуємо центр ребра
                x1, y1 = self.pos[edge[0]]
                x2, y2 = self.pos[edge[1]]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # Створюємо міні-зображення
                imagebox = OffsetImage(image.squeeze(), zoom=0.3, cmap='gray')
                ab = AnnotationBbox(imagebox, center, frameon=False)
                ax.add_artist(ab)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

class IntelligentAgent():
    def __init__(self, start_point, final_point, graph):
        self.start_point = start_point
        self.final_point = final_point
        self.current_point = start_point
        self.graph = graph
        self.graph.node_colors[self.start_point] = 'green'
        self.graph.node_colors[self.final_point] = 'red'
        self.history_moves = []
        self.history = {}
        self.conditional = True
        self.speeds_on_path = []
        self.knowledgeBase = {
            "dead_end": [],
            "history_moves": []
        }
        self.previus_point = start_point
        self.point_state_list = {}
        for key in self.graph.pos.keys():
            self.point_state_list[key] = 0

    def update_position(self, next_node):

        edge_key = (str(self.current_point), str(next_node))
        print(f"Accessing edge: {edge_key}")

        image = self.graph.speeds.get(edge_key)
        if image is not None:
            image_batch = np.expand_dims(image, axis=0)
            prediction = model.predict(image_batch)
            predicted_class = np.argmax(prediction, axis=1)[0]
            print(f"Prediction for edge {edge_key}: {predicted_class}")
        else:
            print(f"No image assigned to edge {edge_key}")


        self.graph.node_colors[self.current_point] = 'blue'
        self.previus_point = self.current_point
        self.current_point = next_node
        self.graph.node_colors[self.current_point] = 'green'
        self.point_state_list[str(self.current_point)] += 1

    def check_node(self, current_point, next_point):
        # перевірка для уникнення постійного "ходіння взад-вперед"
        if next_point == self.previus_point:
            return (current_point in self.knowledgeBase["dead_end"])
        return True

    def move_right(self):
        next_node = self.current_point + self.graph.size
        if ( next_node < self.graph.nodes
             and self.graph.graph.has_edge(f"{self.current_point}", f"{next_node}")
             and next_node not in self.knowledgeBase["dead_end"]
             and self.check_node(self.current_point, next_node) ):
            self.update_position(next_node)
            return True
        return False

    def move_left(self):
        next_node = self.current_point - self.graph.size
        if ( next_node >= 0
             and self.graph.graph.has_edge(f"{self.current_point}", f"{next_node}")
             and next_node not in self.knowledgeBase["dead_end"]
             and self.check_node(self.current_point, next_node) ):
            self.update_position(next_node)
            return True
        return False

    def move_front(self):
        # рух вперед (по x чи y напряму, залежить від self.conditional)
        board_list_for_conditional_true = [(i+1)*self.graph.size - 1 for i in range(self.graph.size)]
        board_list_for_conditional_false = [i*self.graph.size for i in range(1, self.graph.size)]

        if self.conditional:
            # рух направо на сітці
            next_node = self.current_point + 1
            if (self.current_point not in board_list_for_conditional_true
                and next_node < self.graph.nodes
                and self.graph.graph.has_edge(f"{self.current_point}", f"{next_node}")
                and next_node not in self.knowledgeBase["dead_end"]
                and self.check_node(self.current_point, next_node)):
                self.update_position(next_node)
                return True
        else:
            # рух наліво на сітці
            next_node = self.current_point - 1
            if (self.current_point not in board_list_for_conditional_false
                and next_node >= 0
                and self.graph.graph.has_edge(f"{self.current_point}", f"{next_node}")
                and next_node not in self.knowledgeBase["dead_end"]
                and self.check_node(self.current_point, next_node)):
                self.update_position(next_node)
                return True

        return False

    def turn_on_180_deg(self):
        self.conditional = False

    def reverse_front(self):
        self.turn_on_180_deg()
        return self.move_front()

    def random_move(self):
        moves = [self.move_right, self.move_front, self.move_left, self.reverse_front]
        random.shuffle(moves)
        for m in moves:
            if m():
                return True
        return False

    def road_sign(self):
        # позначає глухі кути
        neighbors = list(self.graph.graph.neighbors(f"{self.current_point}"))
        for node in neighbors:
            neighbors_list = list(self.graph.graph.neighbors(node))
            if len(neighbors_list) == 1 and int(node) != self.final_point:
                if int(node) not in self.knowledgeBase["dead_end"]:
                    self.knowledgeBase["dead_end"].append(int(node))

    def check_max_node_state(self):
        # якщо вузол відвідувався більше 4 разів - додаємо до dead_end
        for key in self.point_state_list.keys():
            if self.point_state_list[key] > 10:
                if int(key) not in self.knowledgeBase["dead_end"]:
                    self.knowledgeBase["dead_end"].append(int(key))

    def move_agent(self):
        count = 0
        while(self.current_point != self.final_point and count < 1000):
            self.road_sign()
            self.check_max_node_state()

            # Визначимо напрямок руху до фінальної точки за локальною логікою
            x_final = self.graph.pos[f"{self.final_point}"][0]
            y_final = self.graph.pos[f"{self.final_point}"][1]
            x_current = self.graph.pos[f"{self.current_point}"][0]
            y_current = self.graph.pos[f"{self.current_point}"][1]

            self.moved = False

            # Спробуємо рухатись по X
            if not self.moved:
                if x_final > x_current:
                    # потрібно рухатись вправо
                    self.moved = self.move_right()
                elif x_final < x_current:
                    # потрібно рухатись вліво
                    self.moved = self.move_left()

            # Якщо по X не вдалося, пробуємо по Y
            if not self.moved:
                if y_final > y_current:
                    # рух вперед (conditional = True)
                    self.conditional = True
                    self.moved = self.move_front()
                elif y_final < y_current:
                    # рух назад (conditional = False)
                    self.conditional = False
                    self.moved = self.move_front()

            # Якщо не змогли пересунутись за логікою наближення до цілі, робимо випадковий крок
            if not self.moved:
                self.moved = self.random_move()

            # Якщо ми в глухому куті - позначаємо його
            neighbors_list = list(self.graph.graph.neighbors(f"{self.current_point}"))
            if len(neighbors_list) == 1 and self.current_point != self.final_point:
                if self.current_point not in self.knowledgeBase["dead_end"]:
                    self.knowledgeBase["dead_end"].append(self.current_point)

            end_counter = 0
            for i in neighbors_list:
                if int(i) in self.knowledgeBase["dead_end"]:
                    end_counter += 1
            if end_counter == len(neighbors_list) - 1 and self.current_point != self.final_point:
                if self.current_point not in self.knowledgeBase["dead_end"]:
                    self.knowledgeBase["dead_end"].append(self.current_point)

            self.history[self.current_point] = neighbors_list
            self.knowledgeBase["history_moves"].append(self.current_point)
            count += 1


# Демонстрація
gr = Graph(25)
gr.drawGraph()
gr.removeEdge(15)
gr.assign_speeds()
gr.drawGraph(show_speeds=True)
InAg = IntelligentAgent(2, 24, gr)
gr.drawGraph(show_speeds=True)
InAg.move_agent()
print(InAg.knowledgeBase["dead_end"])
print(InAg.knowledgeBase["history_moves"])
print(gr.speeds_history)
gr.drawGraph(show_speeds=True)

plt.figure(figsize=(12, 5))

# Графік втрат
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Втрата під час тренування та валідації')
plt.xlabel('Епоха')
plt.ylabel('Втрата')
plt.legend()

# Графік точності
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Точність під час тренування та валідації')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()

plt.tight_layout()
plt.show()