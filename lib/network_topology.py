from .base import *

# 网络拓扑属性
class MyGraph():
    def __init__(self, adj_matrix):
        """
        adj_matrix: 邻接矩阵
        """
        #self.adj_matrix = adj_matrix / len(adj_matrix)  # 在每个子网络输入的时候 除以他们的节点数来惩罚
        #self.adj_matrix = scale_one(adj_matrix)
        self.adj_matrix = adj_matrix.copy()

        #self.adj_matrix = adj_matrix / adj_matrix.max().max()
        
        self.init_graph()
        
    def init_graph(self):
        self.g = ig.Graph(directed=True)
        for i in self.adj_matrix.columns:
            self.g.add_vertex(name=i)

        edges = []
        node_labels = self.adj_matrix.columns.values
        weights = []
        for i in (range(self.adj_matrix.shape[0])):
            for j in range(self.adj_matrix.shape[0]):
                if self.adj_matrix.iloc[i, j] != 0 and not np.isnan(self.adj_matrix.iloc[i, j]): 
                    edges.append((i, j))
                    weights.append(self.adj_matrix.iloc[i, j])
        self.g.add_edges(edges)
        self.g.vs['name'] = node_labels
        self.g.vs["label"] = self.g.vs['name']
        self.g.es['edges'] = edges
        self.g.es['weight'] = weights


    def profile(self):
        """
        describe network with key indicators
        """
        #print('stats profile...')
        g = self.g
        weights = self.g.es['weight']
        properties_dict = {
            "节点数": g.vcount(),
            "边数": g.ecount(),
            '是否有向': g.is_directed(),
            '是否加权': g.is_weighted(),
            "节点度数:": g.degree(),
            '最大度': max(g.degree()),
            #'平均度': round(np.sum(self.adj_matrix.values -  np.eye(self.adj_matrix.shape[0], self.adj_matrix.shape[1]) * self.adj_matrix), 5),
            "网络直径": round(g.diameter(weights=weights), 5),  # 
            "平均路径长度": round(g.average_path_length(weights=weights), 5), # 要的
            "度中心性:": g.strength(weights=weights),
            "接近中心性:": g.closeness(weights=weights),    # array/max(array)  
            "中介中心性:": g.betweenness(weights=weights),
            '特征向量中心性': g.evcent(weights=weights),  # 计算特征向量中心性
            '平均介数中心性betweenness': round(np.mean(g.betweenness(weights=weights)), 5),
            '平均接近中心性closeness': round(np.mean(g.closeness(weights=weights)), 5),
            '聚类系数': np.mean(g.transitivity_local_undirected()),
            #'信息熵':  -sum((np.array(g.degree()) / sum(g.degree())) * np.log2((np.array(g.degree()) / sum(g.degree()))))
            #'信息熵': 
        }
        
        return properties_dict


    def plot_adj_matrix_hist(self, figsize=(6, 4)):
        """
        plot histogram of adj-matrix
        """
        pd.DataFrame(self.adj_matrix.values.flatten()).hist(figsize=figsize)

    
    def plot_circle(self, figsize=(6, 4), vertex_color="lightblue"):
        """
        plot network at form of "circle"
        """
        layout = self.g.layout("circle")  #fr
        visual_style = {
            "vertex_size": 20,
            "vertex_color": "blue",
            "vertex_label": self.g.vs["name"],
            "edge_width": 2,
            "layout": layout
        }
        fig, ax = plt.subplots(figsize=figsize)
        ig.plot(self.g, target=ax, **visual_style)
        plt.show()

    
    def plot_clusters(self, figsize=(6,4), vertex_size=0.5, edge_width=0.7):
        """
        plot network at form of "clusters"
        """
        components = self.g.connected_components()
        fig, ax = plt.subplots(figsize=figsize)
        ig.plot(
            components,
            target=ax,
            palette=ig.RainbowPalette(),
            vertex_size=vertex_size,
            vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
            edge_width=edge_width
        )
        plt.show()

    
    def communities_clustering(self):
        """
        使用边介数法进行社区检测
        """
        communities = self.g.community_edge_betweenness()
        clusters = communities.as_clustering()
        print("社区划分:", clusters.membership)

        
        
    # 计算网络信息熵
    def calculate_information_entropy(self):        
        g = self.g
        degrees = g.degree()

        # 计算度的分布（归一化）
        degree_distribution = np.array(degrees) / sum(degrees)

        # 计算信息熵
        entropy = -sum(degree_distribution * np.log2(degree_distribution))

        return entropy
    
    
    def cal_ie(self,df):
        # 计算每个值的出现频率并除以矩阵总元素数以获得概率
        value_counts = df.stack().value_counts(normalize=True)

        # 使用映射函数，将矩阵中的每个值替换为其对应的概率值
        probability_matrix = df.applymap(lambda x: value_counts.get(x))

        # 将结果展示给用户
        #import ace_tools as tools; tools.display_dataframe_to_user(name="Probability Matrix", dataframe=probability_matrix)
        #probability_matrix

        entropy = -(probability_matrix * np.log2(probability_matrix)).sum().sum()

        # 计算归一化熵
        N = df.size  # 网络中的总元素数
        max_entropy = np.log2(N)  # 最大可能熵
        normalized_entropy = entropy / max_entropy  # 归一化熵

        #print("信息熵:", entropy)
        #print("归一化熵:", normalized_entropy)        
        return entropy

    
    
    def calculate_acs(self, df):
        N = df.shape[0]
        acs = df.values.sum() - df.values.diagonal().sum()
        acs = acs / (N -1) / (N-2)
        return acs

    
    # 计算方差分解强度
    def cal_spillover_strength2(self, df):

        np.fill_diagonal(df.values, 0)
        IS = df.sum(axis = 0)
        OS = df.sum(axis =1)
        NS = OS - IS
        #print (NS)
        return IS.values, OS.values, NS.values
    
    
    
    def describe(self):

        # 描述统计
        #for i in df_hz_esg.groupby('年份'):
        #    print (i[-1]['综合评级'].value_counts())
        df_t = pd.DataFrame(df_hz_esg.groupby('年份')['综合评级'].value_counts())
        #.pivot(index='first', columns='second', values='value')
        df_t = df_t.unstack()#.pivot_table()
        df_t.columns = df_t.columns.droplevel(0)


        df_t = df_t[df_t.index >= 2013]
        df_t = df_t[['AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']]
        df_t.to_csv('esg_descrbe.csv')
        vol_interday = vol_interday[(vol_interday.index.year >= 2013) & ((vol_interday.index.year <= 2023))]
        vol_des = vol_interday.describe()

        skewness = vol_interday.skew()
        kurtosis = vol_interday.kurt()
        vol_des.loc['skewness'] = skewness
        vol_des.loc['kurtosis'] = kurtosis


        #df_hz_esg_i['综合评级'].value_counts()

        #df_hz_esg_i = df_hz_esg
        # df_hz_esg_i = df_hz_esg[df_hz_esg['年份'] == year]
        # df_hz_esg_i.sort_values(by = '综合得分')

        # df_hz_esg_i = df_hz_esg_i.loc[df.columns]
        # df_hz_esg_i.loc[:,'rating_score'] = df_hz_esg_i['综合评级']
        #df_hz_esg_i.loc[:,'rating_score'] = pd.qcut(df_hz_esg_i['综合得分'], q = 9, labels= ['AAA', 'AA', 'A', 'BBB', 'BB','B', 'CCC', 'CC','C'][::-1])#.sort_values()
        #df_hz_esg_i.loc[:, 'rating_score'] = pd.qcut(df_hz_esg_i['综合得分'], q=9, labels=['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C'][::-1])

        #df_hz_esg_i['综合评级'].value_counts()[['A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C'][::-1]].plot(kind = 'bar')
        #df_hz_esg_i['综合得分'].plot(kind = 'hist')

        plt.plot(vol_interday[vol_].mean(axis = 1))