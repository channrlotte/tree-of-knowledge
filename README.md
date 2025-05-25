# tree-of-knowledge

## Структура репозитория:

- data
  - texts
    - articles — научные тексты с сайта [КиберЛенинка](https://cyberleninka.ru/)
    - fiction — художественные тексты
  - relations
    - articles — отношения, найденные между словами в научных текстах
    - fiction — отношения, найденные между словами в художественных текстах
  - graphs
    - higher_dim_graphs
      - articles.pickle — граф, построенный на научных текстах
      - fiction.pickle — граф, построенный на художественных текстах
    - clustered_graphs
      - clustered_articles.pickle — кластеризованный граф, построенный на научных текстах
      - clustered_fiction.pickle — кластеризованный граф, построенный на научных текстах
- embeddings
  - cc.ru.100.bin — эмбеддинги [fastText](https://fasttext.cc/docs/en/crawl-vectors.html)
  - dictionary.txt — очищенный [список самых частотных русских слов](https://github.com/hingston/russian/blob/master/100000-russian-words.txt)
- graph
  - vertex.py — описание класса Vertex
  - edge.py — описание класса Edge
  - union_edge.py — описание класса UnionEdge
  - higher_dim_graph.py — описание класса Graph
  - embedding_manager.py — получение эмбеддингов
- get_relations.py — нахождение отношений между словами в тексте
- build_graph.py — построение графа на основе отношений между словами
- visulize_graph.py — визуализация графа
- squeeze_graph.py — сквизинг вершин графа
- metrics.py — подсчёт различных метрик, построение графиков и нахождение хабов
- .gitignore
- requirements.txt
