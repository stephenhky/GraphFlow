import networkx as nx

# words
webpages = ['Big Data 1', 'Big Data 2', 'Machine Learning 1', 'Artificial Intelligence', 'Deep Learning 1',
            'Machine Learning 2', 'Deep Learning 2', 'Big Data 3',
            'Deep Learning 3', 'Econophysics', 'Dow-Jones 1', 'Wall Street', 'Hadoop', 'Spark', 'Dow-Jones 2',
            'Big Data Fake 1', 'Big Data Fake 2', 'Porn 1', 'Porn 2']
links = [('Big Data 1', 'Big Data 2'), ('Big Data 2', 'Big Data 1'), ('Big Data 3', 'Big Data 2'),
         ('Big Data 3', 'Big Data 1'), ('Big Data 3', 'Deep Learning 1'),
         ('Machine Learning 1', 'Artificial Intelligence'), ('Deep Learning 1', 'Artificial Intelligence'),
    ('Deep Learning 1', 'Machine Learning 1'), ('Deep Learning 2', 'Machine Learning 1'),
    ('Deep Learning 1', 'Big Data 1'), ('Big Data 1', 'Deep Learning 1'), ('Big Data 2', 'Deep Learning 1'),
    ('Big Data 3', 'Big Data 1'), ('Big Data 2', 'Econophysics'), ('Econophysics', 'Big Data 2'),
    ('Econophysics', 'Dow-Jones 1'), ('Big Data 2', 'Dow-Jones 1'), ('Big Data 2', 'Dow-Jones 2'), ('Big Data 1', 'Hadoop'),
    ('Big Data 2', 'Hadoop'), ('Big Data 3', 'Hadoop'), ('Big Data 1', 'Spark'), ('Spark', 'Hadoop'), ('Hadoop', 'Spark'),
    ('Hadoop', 'Big Data 1'), ('Spark', 'Big Data 1'), ('Wall Street', 'Big Data 2'), ('Wall Street', 'Spark'),
    ('Dow-Jones 2', 'Dow-Jones 1'), ('Dow-Jones 2', 'Big Data 1'), ('Big Data Fake 1', 'Porn 1'), ('Big Data Fake 2', 'Big Data Fake 1'),
    ('Big Data Fake 2', 'Porn 1'), ('Porn 1', 'Porn 2'), ('Machine Learning 2', 'Machine Learning 1'), ('Machine Learning 2', 'Big Data 1'),
         ('Deep Learning 3', 'Deep Learning 1'), ('Machine Learning 2', 'Artificial Intelligence'), ('Deep Learning 1', 'Machine Learning 2'),
    ('Econophysics', 'Big Data 1'), ('Dow-Jones 2', 'Big Data 1'), ('Big Data 1', 'Dow-Jones 2'), ('Big Data 1', 'Deep Learning 2'),
    ('Porn 1', 'Big Data 3')]

webnet = nx.DiGraph()
webnet.add_nodes_from(webpages)
webnet.add_edges_from(links)