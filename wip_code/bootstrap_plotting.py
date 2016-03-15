import seaborn as sns
from seaborn import plt
import networkx as nx

from matplotlib import rc
# rc('text', usetex=True)

font = {'size' : 8}
axes = {'labelsize': 8, 'titlesize' : 8}
rc('font', **font)  
rc('axes', **axes)


def make_plots(result_dict, filename="Bootstrap_Results.png"):
    
    sns.set_style('ticks')
    
    fig, axarr = plt.subplots(4, sharex=True,figsize=(16,10))
    axarr[0].set_xlim(-4.2, 3.2)
    arrowprops=dict(arrowstyle="->",connectionstyle="arc3")

    
    axarr[0].set_ylabel(r"Bias measured log D")    
    axarr[1].set_ylabel(r"Stdev log D")
    axarr[2].set_ylabel(r"CV cyclohexane signal")
    axarr[3].set_ylabel(r"CV buffer signal")
    axarr[3].set_xlabel(r"Measured Log D")
    
    bias_labels = label_positions(result_dict, "bias")
    std_labels = label_positions(result_dict, "stdev")
    chx_labels= label_positions(result_dict, "chx_CV")
    buf_labels= label_positions(result_dict, "buf_CV")
    
    for compound,dp in result_dict.items():
        ev, b, s, xcv, bcv = dp["expected_value"], dp["bias"], dp["stdev"], dp["chx_CV"], dp["buf_CV"]
            
        axarr[0].annotate(compound.replace('SAMPL5_', ''), xy=[ev,b], xytext=bias_labels[compound], arrowprops=arrowprops)            
        axarr[0].scatter(ev,b, color="g")
        
        axarr[1].annotate(compound.replace('SAMPL5_', ''), xy=[ev,s], xytext=std_labels[compound], arrowprops=arrowprops)
        axarr[1].scatter(ev, s, color='r' )
        
        axarr[2].annotate(compound.replace('SAMPL5_', ''), xy=[ev,xcv], xytext=chx_labels[compound], arrowprops=arrowprops)
        axarr[2].scatter(ev,xcv, color='b' )
        
        axarr[3].annotate(compound.replace('SAMPL5_', ''), xy=[ev,bcv], xytext=buf_labels[compound], arrowprops=arrowprops)
        axarr[3].scatter(ev, bcv, color='k' )
    
    # Add some y-spacing for labels
    for ax in axarr:
        limits = ax.get_ylim()
        rng = limits[1] - limits[0]
        ax.set_ylim([limits[0]-0.175*rng, limits[1]+0.175*rng])
    
    fig.savefig(filename)

def label_positions(result_dict, prop, edge_length=0.085):
    
    G=nx.Graph()        
    fixed_positions = dict()
    
    for comp, datapoints in result_dict.items():
        G.add_node(comp + "data")
        G.add_node(comp)
        G.add_edge(comp + "data", comp)
        fixed_positions[comp +"data"] = [datapoints["expected_value"], datapoints[prop]]
   
    pos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_positions.keys(), k=edge_length)
    # nx.draw_networkx(G,pos)
    
    return {k: v for k, v in pos.items() if "data" not in k}