# import numpy as np
#
# from edunet.core import FlowGraph, Input, FlowControl
# from edunet.core.operations import Dense, SquaredDistance
#
#
# def test():
#     def create_algebra_dataset(n: int, seed: int = None):
#         rand = np.random.RandomState(seed)
#         x1 = rand.uniform(size=(n, 1))
#         x2 = rand.uniform(size=(n, 1))
#         y = x1 / 3. + x2 / 7. - 2.
#         return x1, x2, y
#
#     X1, X2, Y = create_algebra_dataset(4)
#
#     i = 0
#     x = np.array([X1[i], X2[i]], np.float32)
#     y = np.float32(np.reshape(Y[i], (1, 1)))
#
#     graph = FlowGraph()
#
#     input_data = graph.add(Input((2, 1), np.float32))
#     input_labels = graph.add(Input((1, 1), np.float32))
#
#     dense_1 = graph.add(Dense(input_data, 1))
#
#     loss = graph.add(SquaredDistance(dense_1, input_labels))
#
#     flow = FlowControl(graph)
#
#     flow.feed({
#         input_data: x,
#         input_labels: y,
#     })
#
#     flow.run_forward()
#     flow.run_backward()
#
#
# test()
