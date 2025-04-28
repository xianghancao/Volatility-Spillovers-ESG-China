import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm



def plot_test():
    

    import networkx as nx
    import matplotlib.pyplot as plt

    # 创建一个随机图，使用NetworkX的Barabási-Albert模型
    G = nx.barabasi_albert_graph(n=100, m=3)

    # 计算节点的中心性
    centrality = nx.betweenness_centrality(G)

    # 设置节点的大小和颜色（根据中心性值）
    node_size = [v * 10000 for v in centrality.values()]
    node_color = list(centrality.values())

    # 绘制图形
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(
        G,
        node_size=node_size,
        node_color=node_color,
        cmap=plt.cm.viridis,
        with_labels=False,
        edge_color='gray',
        alpha=0.7,
        pos=nx.spring_layout(G, k=0.1, iterations=20)
    )

    plt.title("Complex Network Visualization")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="Centrality")
    plt.show()

    
    
def plot_test2():


    # 读取边和节点数据
    #edges_df = pd.read_csv('edges.csv')  # 假设数据存储在edges.csv中
    #nodes_df = pd.read_csv('nodes.csv')  # 假设数据存储在nodes.csv中

    # 创建一个空的NetworkX图
    G = nx.Graph()

    # 向图中添加节点
    for _, row in nodes.iterrows():
        G.add_node(row['ID'], label=row['Label'], group=row['Group'])

    # 向图中添加边
    for _, row in edges_sparse.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], group=row['Group'])

    # 创建评级到颜色的映射
    group_color_map = {
        'AAA': 'green',
        'AA': 'blue',
        'A': 'cyan',
        'BBB': 'orange',
        'BB': 'red',
        'B': 'purple',
        'CCC': 'brown',
        'CC': 'pink',
        'C': 'gray',
        'D': 'black'
    }

    # 将节点的Group属性转换为颜色
    node_color = [group_color_map[G.nodes[node]['group']] for node in G.nodes]
    node_size = [G.degree(node) * 10 for node in G.nodes]

    # 绘制图形
    plt.figure(figsize=(14, 14))

    # 使用spring布局绘图
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)

    # 绘制边
    edges = nx.draw_networkx_edges(
        G, pos, 
        edge_color=[G[u][v]['weight'] for u, v in G.edges], 
        edge_cmap=plt.cm.Blues, 
        edge_vmin=min(edges_sparse['Weight']), 
        edge_vmax=max(edges_sparse['Weight']),
        alpha=0.5
    )

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, {node: G.nodes[node]['label'] for node in G.nodes}, font_size=8)

    # 显示图形
    plt.title("Complex Network Visualization")
    plt.colorbar(edges, label='Edge Weight')
    plt.show()




def formor_():
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    G = nx.Graph()

    group_map = {'AAA': 'AAA,AA,A', 'AA': 'AAA,AA,A', 'A': 'AAA,AA,A', 'BBB': 'BBB', 'BB': 'BB', 'B': 'B', 
                 'CCC': 'CCC', 'CC': 'CC,C', 'C': 'CC,C'}

    # 向图中添加节点，
    for _, row in nodes.iterrows():
        if row['Group'] != 'D':
            G.add_node(row['ID'], label=row['Label'], group=group_map.get(row['Group'], row['Group']))

    # 向图中添加边
    for _, row in edges_sparse.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], group=row['Group'])

    group_color_map = {
        'AAA,AA,A': 'green',
        'BBB': 'orange',
        'BB': 'red',
        'B': 'purple',
        'CCC': 'brown',
        'CC,C': 'pink',
    }

    node_color = [group_color_map[G.nodes[node]['group']] for node in G.nodes]

    node_size = [G.degree(node) * 0.5 for node in G.nodes]

    pos = nx.spring_layout(G, k=0.5, iterations=100)

    for group in set(group_color_map.keys()):
        # 找出属于当前group的节点
        group_nodes = [node for node in G.nodes if G.nodes[node]['group'] == group]
        # 计算这些节点的重心
        x_mean = sum([pos[node][0] for node in group_nodes]) / len(group_nodes)
        y_mean = sum([pos[node][1] for node in group_nodes]) / len(group_nodes)
        # 将这些节点往重心方向微调
        for node in group_nodes:
            pos[node][0] += (x_mean - pos[node][0]) * 0.1
            pos[node][1] += (y_mean - pos[node][1]) * 0.1

    plt.figure(figsize=(15, 10))

    # 绘制边，根据权重调整边的宽度和颜色
    edges = nx.draw_networkx_edges(G, pos,
                                   edge_color=[G[u][v]['weight']*3 for u, v in G.edges()],
                                   width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
                                   edge_cmap=plt.cm.Blues,
                                   alpha=0.7)

    # 绘制节点，使用group颜色
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes,
                           node_size=node_size,
                           node_color=node_color,
                           alpha=0.8)

    # 添加图例
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=group)
                      for group, color in group_color_map.items()]
    plt.legend(handles=legend_handles, title="Node Groups", loc='upper right')

    # 调整xlim和ylim以显示完整网络
    plt.xlim(min([x for x, y in pos.values()]) - 0.1, max([x for x, y in pos.values()]) + 0.1)
    plt.ylim(min([y for x, y in pos.values()]) - 0.1, max([y for x, y in pos.values()]) + 0.1)

    plt.colorbar(edges, label='Edge Weight')

    # plt.axis('off')
    plt.show()




def plot_1():
    # 创建一个空的NetworkX图
    G = nx.Graph()

    # 合并 AAA, AA, A 组，删除 D 组
    group_map = {'AAA': 'AAA,AA,A', 'AA': 'AAA,AA,A', 'A': 'AAA,AA,A', 'BBB': 'BBB', 'BB': 'BB', 'B': 'B', 
                 'CCC': 'CCC', 'CC': 'CC,C', 'C': 'CC,C'}

    # 向图中添加节点，过滤掉 D 组
    for _, row in nodes.iterrows():
        if row['Group'] != 'D':
            G.add_node(row['ID'], label=row['Label'], group=group_map.get(row['Group'], row['Group']))

    # 向图中添加边
    for _, row in edges_sparse.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], group=row['Group'])

    # 创建评级到颜色的映射
    group_color_map = {
        'AAA,AA,A': 'green',
        'BBB': 'orange',
        'BB': 'red',
        'B': 'purple',
        'CCC': 'brown',
        'CC,C': 'pink',
    }

    # 根据group设置节点颜色
    node_color = [group_color_map[G.nodes[node]['group']] for node in G.nodes]

    # 根据连接强度（度数）设置节点大小，整体缩小节点大小
    node_size = [G.degree(node) * 0.2 for node in G.nodes]

    # 使用spring布局来绘制图形
    pos = nx.spring_layout(G, k=0.5, iterations=100)

    # 微调同一组节点的位置，使它们更集中
    for group in set(group_color_map.keys()):
        # 找出属于当前group的节点
        group_nodes = [node for node in G.nodes if G.nodes[node]['group'] == group]
        # 计算这些节点的重心
        x_mean = sum([pos[node][0] for node in group_nodes]) / len(group_nodes)
        y_mean = sum([pos[node][1] for node in group_nodes]) / len(group_nodes)
        # 将这些节点往重心方向微调
        for node in group_nodes:
            pos[node][0] += (x_mean - pos[node][0]) * 0.1
            pos[node][1] += (y_mean - pos[node][1]) * 0.1

    plt.figure(figsize=(25, 25))

    # 绘制边，根据权重调整边的宽度和颜色，颜色加深
    edges = nx.draw_networkx_edges(G, pos,
                                   edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                                   width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
                                   edge_cmap=plt.cm.Blues,
                                   edge_vmin=min([G[u][v]['weight'] for u, v in G.edges()]),
                                   edge_vmax=max([G[u][v]['weight'] for u, v in G.edges()]),
                                   alpha=0.9)  # 加深边的颜色

    # 绘制节点，使用group颜色
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes,
                           node_size=node_size,
                           node_color=node_color,
                           alpha=0.8)

    # 添加图例
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=group)
                      for group, color in group_color_map.items()]
    plt.legend(handles=legend_handles, title="Node Groups", loc='upper right')

    # 调整xlim和ylim以显示完整网络
    plt.xlim(min([x for x, y in pos.values()]) - 0.1, max([x for x, y in pos.values()]) + 0.1)
    plt.ylim(min([y for x, y in pos.values()]) - 0.1, max([y for x, y in pos.values()]) + 0.1)

    plt.colorbar(edges, label='Edge Weight')

    # plt.axis('off')
    plt.savefig('pics/net.jpg')

    
    
# 第二个版本
def plot_2():

    G = nx.Graph()

    group_map = {'AAA': 'AAA,AA,A', 'AA': 'AAA,AA,A', 'A': 'AAA,AA,A', 'BBB': 'BBB', 'BB': 'BB', 'B': 'B', 
                 'CCC': 'CCC', 'CC': 'CC,C', 'C': 'CC,C'}

    # 向图中添加节点
    for _, row in nodes.iterrows():
        if row['Group'] != 'D':
            G.add_node(row['ID'], label=row['Label'], group=group_map.get(row['Group'], row['Group']))

    # 向图中添加边
    for _, row in edges_sparse.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], group=row['Group'])

    group_color_map = {
        'AAA,AA,A': 'green',
        'BBB': 'orange',
        'BB': 'red',
        'B': 'purple',
        'CCC': 'brown',
        'CC,C': 'pink',
    }


    node_color = [group_color_map[G.nodes[node]['group']] for node in G.nodes]

    node_size = [G.degree(node) * 0.2 for node in G.nodes]

    pos = nx.spring_layout(G, k=0.5, iterations=100)

    # 微调同一组节点的位置，使它们更集中
    for group in set(group_color_map.keys()):
        group_nodes = [node for node in G.nodes if G.nodes[node]['group'] == group]
        x_mean = sum([pos[node][0] for node in group_nodes]) / len(group_nodes)
        y_mean = sum([pos[node][1] for node in group_nodes]) / len(group_nodes)
        for node in group_nodes:
            pos[node][0] += (x_mean - pos[node][0]) * 0.1
            pos[node][1] += (y_mean - pos[node][1]) * 0.1

    plt.figure(figsize=(25, 25))

    edges = nx.draw_networkx_edges(G, pos,
                                   edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                                   width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
                                   edge_cmap=plt.cm.Blues,
                                   edge_vmin=min([G[u][v]['weight'] for u, v in G.edges()]),
                                   edge_vmax=max([G[u][v]['weight'] for u, v in G.edges()]),
                                   alpha=0.9)  # 加深边的颜色

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes,
                           node_size=node_size,
                           node_color=node_color,
                           alpha=0.8)

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=group)
                      for group, color in group_color_map.items()]
    plt.legend(handles=legend_handles, title="Node Groups", loc='upper right')

    plt.xlim(min([x for x, y in pos.values()]) - 0.1, max([x for x, y in pos.values()]) + 0.1)
    plt.ylim(min([y for x, y in pos.values()]) - 0.1, max([y for x, y in pos.values()]) + 0.1)

    plt.colorbar(edges, label='Edge Weight')

    plt.savefig('pics/net.jpg')

    
    
    
    



def plot_network(nodes, edges_sparse, month_ = '2020-01-31'):
    # 创建一个空的NetworkX图
    G = nx.Graph()

    # 合并 AAA, AA, A 组，删除 D 组
    group_map = {'AAA': 'AAA,AA,A', 'AA': 'AAA,AA,A', 'A': 'AAA,AA,A', 'BBB': 'BBB', 'BB': 'BB', 'B': 'B', 
                 'CCC': 'CCC', 'CC': 'CC,C', 'C': 'CC,C'}

    for _, row in nodes.iterrows():
        if row['Group'] != 'D':
            G.add_node(row['ID'], label=row['Label'], group=group_map.get(row['Group'], row['Group']))

    for _, row in edges_sparse.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], group=row['Group'])

    group_color_map = {
        'AAA,AA,A': 'green',
        'BBB': 'orange',
        'BB': 'red',
        'B': 'purple',
        'CCC': 'brown',
        'CC,C': 'pink',
    }

    node_color = [group_color_map[G.nodes[node]['group']] for node in G.nodes]

    node_size = [G.degree(node) * 0.8 for node in G.nodes]

    pos = nx.spring_layout(G, k=0.7, iterations=100)  # k值适中，节点分布适度松散

    group_offsets = {
        'AAA,AA,A': (0, 0),
        'BBB': (2, 2),
        'BB': (-2, 2),
        'B': (2, -2),
        'CCC': (-2, -2),
        'CC,C': (3, 0),
    }

    for group, offset in group_offsets.items():
        group_nodes = [node for node in G.nodes if G.nodes[node]['group'] == group]
        x_mean = sum([pos[node][0] for node in group_nodes]) / len(group_nodes)
        y_mean = sum([pos[node][1] for node in group_nodes]) / len(group_nodes)
        for node in group_nodes:
            pos[node][0] += (x_mean - pos[node][0]) * 0.05 + offset[0] * 0.5
            pos[node][1] += (y_mean - pos[node][1]) * 0.05 + offset[1] * 0.5

    plt.figure(figsize=(15, 20))  # 增大图形尺寸

    edges = nx.draw_networkx_edges(G, pos,
                                   edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                                   width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
                                   edge_cmap=plt.cm.Blues,
                                   edge_vmin=min([G[u][v]['weight'] for u, v in G.edges()]),
                                   edge_vmax=max([G[u][v]['weight'] for u, v in G.edges()]),
                                   alpha=0.9)  # 加深边的颜色

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes,
                           node_size=node_size,
                           node_color=node_color,
                           alpha=0.8)

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=group)
                      for group, color in group_color_map.items()]
    plt.legend(handles=legend_handles, title="Node Groups", loc='upper right')

    plt.xlim(min([x for x, y in pos.values()]) - 0.1, max([x for x, y in pos.values()]) + 0.1)
    plt.ylim(min([y for x, y in pos.values()]) - 0.1, max([y for x, y in pos.values()]) + 0.1)

    plt.colorbar(edges, label='Edge Weight')

    plt.savefig('pics/net_%s.jpg'%month_)

nodes, edges_sparse = get_node_edge()
#plot_network(nodes, edges_sparse)






def plot_network_2(nodes, edges_sparse, month_):

    G = nx.Graph()

    group_map = {'AAA': 'AAA,AA,A', 'AA': 'AAA,AA,A', 'A': 'AAA,AA,A', 'BBB': 'BBB', 'BB': 'BB', 'B': 'B', 
                 'CCC': 'CCC', 'CC': 'CC,C', 'C': 'CC,C'}

    # 向图中添加节点
    for _, row in nodes.iterrows():
        if row['Group'] != 'D':
            G.add_node(row['ID'], label=row['Label'], group=group_map.get(row['Group'], row['Group']))

    for _, row in edges_sparse.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], group=row['Group'])

    group_color_map = {
        'AAA,AA,A': 'green',
        'BBB': 'blue',
        'BB': 'gray',
        'B': 'purple',
        'CCC': 'red',
        'CC,C': 'black',
    }

    node_color = [group_color_map[G.nodes[node]['group']] for node in G.nodes]

    node_size = [G.degree(node) * 1.6 for node in G.nodes]

    pos = nx.spring_layout(G, k=1.2, iterations=100)  # 增大k值，节点分布更广

    group_offsets = {
        'AAA,AA,A': (0, 0),
        'BBB': (1, 1),
        'BB': (-1, 1),
        'B': (1, -1),
        'CCC': (-1, -1),
        'CC,C': (1.5, 0),
    }

    for group, offset in group_offsets.items():
        group_nodes = [node for node in G.nodes if G.nodes[node]['group'] == group]
        x_mean = sum([pos[node][0] for node in group_nodes]) / len(group_nodes)
        y_mean = sum([pos[node][1] for node in group_nodes]) / len(group_nodes)
        for node in group_nodes:
            pos[node][0] += (x_mean - pos[node][0]) * 0.03 + offset[0] * 0.3
            pos[node][1] += (y_mean - pos[node][1]) * 0.03 + offset[1] * 0.3

    plt.figure(figsize=(25, 20)) 

    edges = nx.draw_networkx_edges(G, pos,
                                   edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                                   width=[G[u][v]['weight'] * 10 for u, v in G.edges()],
                                   edge_cmap=plt.cm.Blues,
                                   edge_vmin=min([G[u][v]['weight'] for u, v in G.edges()]),
                                   edge_vmax=max([G[u][v]['weight'] for u, v in G.edges()]),
                                   alpha=0.9)  # 加深边的颜色

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes,
                           node_size=node_size,
                           node_color=node_color,
                           alpha=0.8)

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=group)
                      for group, color in group_color_map.items()]
    plt.legend(handles=legend_handles, title="Node Groups", loc='upper right')

    plt.xlim(min([x for x, y in pos.values()]) - 0.1, max([x for x, y in pos.values()]) + 0.1)
    plt.ylim(min([y for x, y in pos.values()]) - 0.1, max([y for x, y in pos.values()]) + 0.1)

    plt.colorbar(edges, label='Edge Weight')

    plt.savefig('pics/net_%s.jpg'%month_)

    
    
    
def plot_3():
    
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # 创建一个空的NetworkX图
    G = nx.Graph()

    # 合并 AAA, AA, A 组，删除 D 组
    group_map = {'AAA': 'AAA,AA,A', 'AA': 'AAA,AA,A', 'A': 'AAA,AA,A', 'BBB': 'BBB', 'BB': 'BB', 'B': 'B', 
                 'CCC': 'CCC', 'CC': 'CC', 'C': 'C'}

    # 向图中添加节点，过滤掉 D 组
    for _, row in nodes.iterrows():
        if row['Group'] != 'D':
            G.add_node(row['ID'], label=row['Label'], group=group_map.get(row['Group'], row['Group']))

    # 向图中添加边
    for _, row in edges_sparse.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'], group=row['Group'])

    # 创建评级到颜色的映射
    group_color_map = {
        'AAA,AA,A': 'green',
        'BBB': 'orange',
        'BB': 'red',
        'B': 'purple',
        'CCC': 'brown',
        'CC': 'pink',
        'C': 'gray'
    }

    # 根据group设置节点颜色
    node_color = [group_color_map[G.nodes[node]['group']] for node in G.nodes]

    # 根据连接强度（度数）设置节点大小，整体缩小节点大小
    node_size = [G.degree(node) * 0.5 for node in G.nodes]

    # 使用spring布局来绘制图形
    pos = nx.spring_layout(G, k=0.5, iterations=100)

    # 微调同一组节点的位置，使它们更集中
    for group in set(group_color_map.keys()):
        # 找出属于当前group的节点
        group_nodes = [node for node in G.nodes if G.nodes[node]['group'] == group]
        # 计算这些节点的重心
        x_mean = sum([pos[node][0] for node in group_nodes]) / len(group_nodes)
        y_mean = sum([pos[node][1] for node in group_nodes]) / len(group_nodes)
        # 将这些节点往重心方向微调
        for node in group_nodes:
            pos[node][0] += (x_mean - pos[node][0]) * 0.1
            pos[node][1] += (y_mean - pos[node][1]) * 0.1

    plt.figure(figsize=(25, 25))

    # 绘制边，根据权重调整边的宽度和颜色
    edges = nx.draw_networkx_edges(G, pos,
                                   edge_color=[G[u][v]['weight']*30 for u, v in G.edges()],
                                   width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
                                   edge_cmap=plt.cm.Blues,
                                   alpha=0.7)

    # 绘制节点，使用group颜色
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes,
                           node_size=node_size,
                           node_color=node_color,
                           alpha=0.8)

    # 添加图例
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=group)
                      for group, color in group_color_map.items()]
    plt.legend(handles=legend_handles, title="Node Groups", loc='upper right')

    # 调整xlim和ylim以显示完整网络
    plt.xlim(min([x for x, y in pos.values()]) - 0.1, max([x for x, y in pos.values()]) + 0.1)
    plt.ylim(min([y for x, y in pos.values()]) - 0.1, max([y for x, y in pos.values()]) + 0.1)

    plt.colorbar(edges, label='Edge Weight')

    # plt.axis('off')
    plt.show()
