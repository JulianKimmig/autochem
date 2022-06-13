from typing import List, Tuple

import networkx as nx
import matplotlib.pyplot as plt

COLORTYPE={
    "reactant": "red",
    "reaction": "green",
}

class ReactionGraph():
    def add_reactant(self, reactant):
        self.add_node(reactant)
        self.nodes[reactant]['ntype'] = "reactant"
        self.set_initial_concentration(reactant, 0)

    def set_initial_concentration(self, reactant, conc):
        if reactant not in self.nodes:
            self.add_reactant(reactant)
        self.nodes[reactant]['conc'] = conc

    def display(self):
        pos = nx.kamada_kawai_layout(self)

        nx.draw_networkx_nodes(self, pos, node_color = [COLORTYPE[self.nodes[n]["ntype"]] for n in self.nodes], node_size = 100, alpha = 1)


        curved_edges = [edge for edge in self.edges() if (edge[1],edge[0]) in self.edges()]
        straight_edges = list(set(self.edges()) - set(curved_edges))
        #ax = plt.gca()
        nx.draw_networkx_edges(self, pos, edgelist=straight_edges)
        arc_rad = 0.1
        print(curved_edges)
        print(pos)
        nx.draw_networkx_edges(self, pos, edgelist=curved_edges, connectionstyle="arc3,rad=0.1")


        plt.show()
        plt.close()

class ReactionGraph1(ReactionGraph,nx.MultiGraph):
    def add_reaction(self,name, educts:List[Tuple[int,str]], products:List[Tuple[int,str]], ks:Tuple[float,float]):

        for n,educt in educts:
            if educt not in self.nodes:
                self.add_reactant(educt)
        for n,product in products:
            if product not in self.nodes:
                self.add_reactant(product)

        for n,educt in educts:
            for m,product in products:
                self.add_edge(educt,product,name=name,k=ks[0],rxn=name)
                self.add_edge(product,educt,name=name,k=ks[1],rxn=name)

class ReactionGraph2(ReactionGraph,nx.DiGraph):

    def add_reaction(self,name, educts:List[Tuple[int,str]], products:List[Tuple[int,str]], ks:Tuple[float,float]):

        self.add_node(name)
        self.nodes[name]['ntype'] = "reaction"
        self.nodes[name]['k1'] = ks[0]
        self.nodes[name]['k2'] = ks[1]

        for n,educt in educts:
            if educt not in self.nodes:
                self.add_reactant(educt)
            self.add_edge(educt, name,count=n,k=ks[0])
            self.add_edge(name,educt,count=n,k=ks[1])
        for n,product in products:
            if product not in self.nodes:
                self.add_reactant(product)
            self.add_edge(product, name,count=n,k=ks[0])
            self.add_edge(name,product,count=n,k=ks[1])

if __name__ == "__main__":
    g = ReactionGraph1()
    g.add_reaction("R1", [(2,"A")], [(1,"B")], (1,0))
    g.add_reaction("R2", [(1,"B")], [(1,"C")], (1,1))
    g.add_reaction("R3", [(1,"C")], [(2,"A")], (1,0))
    g.display()