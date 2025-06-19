/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package agn;

import charts.Chart;
import charts.GraphDR;
// import charts.PrefuseFrame;
import com.jgraph.layout.JGraphFacade;
import com.jgraph.layout.JGraphLayout;
import com.jgraph.layout.graph.JGraphSimpleLayout;
import com.jgraph.layout.organic.JGraphOrganicLayout;
import fs.FS;
import fs.Preprocessing;
import java.awt.Color;
import java.awt.geom.Rectangle2D;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Vector;
import javax.swing.JOptionPane;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jgraph.JGraph;
import org.jgraph.graph.AttributeMap;
import org.jgraph.graph.DefaultCellViewFactory;
import org.jgraph.graph.DefaultEdge;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.DefaultGraphModel;
import org.jgraph.graph.GraphConstants;
import org.jgraph.graph.GraphLayoutCache;
import org.jgraph.graph.GraphModel;
import prefuse.Visualization;
import prefuse.visual.NodeItem;
import utilities.IOFile;
import utilities.MathRoutines;
import utilities.Timer;

/**
 * @author Fabricio
 */
public class AGNRoutines {

    public static Timer timer = new Timer();

    /* Metodo usado para ler um conjunto de regras (genes preditores, seus
    estados e inferir estado do gene target.*/

    public static boolean RefreshGraphPositionsonAGN(AGN agn, JGraph graph) {
        Object[] vertices = graph.getGraphLayoutCache().getCells(false, true, false, false);
        for (int i = 0; i < vertices.length; i++) {
            DefaultGraphCell node = (DefaultGraphCell) vertices[i];
            Map attr = node.getAttributes();
            Rectangle2D rect = GraphConstants.getBounds(attr);
            //laco para cada vertice do grafo
            int geneindex = (Integer) attr.get("index");
            agn.getGenes()[geneindex].setX((float) rect.getX());
            agn.getGenes()[geneindex].setY((float) rect.getY());
        }
        return (true);
    }

    public static boolean RefreshGraphPositionsonAGN(AGN agn, Visualization vis) {
        // Iterator nodes = vis.items(PrefuseFrame.NODES);
        Iterator nodes = vis.items("graph.nodes");
        while (nodes.hasNext()) {
            NodeItem node = (NodeItem) nodes.next();
            int id = node.getInt("id");
            float x = (float) node.getX();
            float y = (float) node.getY();
            Gene gene = agn.getGenes()[id];
            gene.setX(x);
            gene.setY(y);
        }
        return (true);
    }

    public static GraphDR ViewAGNMA(AGN agn, Vector targetindexes) {
        boolean dolayout = false;
        // Construct Model and GraphDR
        GraphModel model = new DefaultGraphModel();
        GraphLayoutCache view = new GraphLayoutCache(model, new DefaultCellViewFactory());
        JGraph graph = new JGraph(model, view);

        //JGraph graph = new JGraph(model);
        // Control-drag should clone selection
        graph.setCloneable(true);
        // Enable edit without final RETURN keystroke
        graph.setInvokesStopCellEditing(true);
        // When over a cell, jump to its default port (we only
        // have one, anyway)
        graph.setJumpToDefaultPort(true);
        // Insert all three cells in one call, so we need an
        // array to store them
        List<DefaultGraphCell> vertex = new ArrayList<DefaultGraphCell>();
        List<DefaultGraphCell> edges = new ArrayList<DefaultGraphCell>();
        //List<DefaultGraphCell> blueedges = new ArrayList<DefaultGraphCell>();
        //List<DefaultGraphCell> rededges = new ArrayList<DefaultGraphCell>();
        //List<DefaultGraphCell> orangeedges = new ArrayList<DefaultGraphCell>();

        double rX = (GraphDR.w - 100) / 2;
        double rY = (GraphDR.h - 100) / 2;
        double theta = 0;
        double delta = 2 * Math.PI / agn.getGenes().length;

        int count = 0;
        for (int gt = 0; gt < agn.getGenes().length; gt++) {
            String name = "g" + String.valueOf(gt);
            if (agn.getGenes()[gt].getName() != null) {
                name = agn.getGenes()[gt].getName();// + "(" + String.valueOf(gt) + ")";
            }
            float x = agn.getGenes()[gt].getX();
            float y = agn.getGenes()[gt].getY();
            if (x == 0 && y == 0) {//atribui posicoes iniciais para os genes (anel circular)
                x = (float) (rX + 10 + (rX * Math.cos(theta)));
                y = (float) (rY + 10 + (rY * Math.sin(theta)));
                agn.getGenes()[gt].setX(x);
                agn.getGenes()[gt].setY(y);
                count++;
            }
            theta += delta;
            DefaultGraphCell node = GraphDR.createVertex(name, x, y, 0, 0, Color.white, false, true);
            //node.setUserObject(agn.getGenes()[gt].getProbsetname());
            vertex.add(node);
            Color cor = MainMarieAnne.getBGColor(agn.getGenes()[gt]);
            GraphConstants.setGradientColor(vertex.get(gt).getAttributes(), cor);
        }
        if (count > 0.5 * agn.getNrgenes()) {//mais da metade dos genes da rede esta com coordenada 0,0
            dolayout = true;
        }
        Vector nodes = new Vector();
        for (int gt = 0; gt < agn.getGenes().length; gt++) {
            for (int gp = 0; gp < agn.getGenes()[gt].getPredictors().size(); gp++) {
                if (!nodes.contains(gt)) {
                    nodes.add(gt);
                }
                int predictor = (Integer) agn.getGenes()[gt].getPredictors().get(gp);
                if (!nodes.contains(predictor)) {
                    nodes.add(predictor);
                }
                DefaultEdge edge;
                if (agn.getGenes()[gt].getCfvalues().size() > gp) {
                    edge = new DefaultEdge(agn.getGenes()[gt].getCfvalues().get(gp));
                } else {
                    edge = new DefaultEdge();
                }
                edge.setSource(vertex.get(predictor).getChildAt(0));
                edge.setTarget(vertex.get(gt).getChildAt(0));
                GraphConstants.setLineEnd(edge.getAttributes(), GraphConstants.ARROW_SIMPLE);
                GraphConstants.setRouting(edge.getAttributes(), GraphConstants.ROUTING_DEFAULT);
                GraphConstants.setLineStyle(edge.getAttributes(), GraphConstants.STYLE_SPLINE);
                GraphConstants.setLineWidth(edge.getAttributes(), 2);
                GraphConstants.setLineColor(edge.getAttributes(), Color.BLACK);
                edges.add(edge);
            }
        }

        for (int i = 0; i < vertex.size(); i++) {
            if (nodes.contains(i) || (targetindexes != null && targetindexes.contains(i))) {
                DefaultGraphCell node = vertex.get(i);
                AttributeMap attr = node.getAttributes();
                //Rectangle2D rect = GraphConstants.getBounds(attr);
                //float x = agn.getGenes()[i].getX();
                //float y = agn.getGenes()[i].getY();
                //rect.setRect(x, y, rect.getWidth(), rect.getHeight());
                //GraphConstants.setBounds(attr, rect);
                GraphConstants.setAutoSize(attr, true);
                GraphConstants.setConstrained(attr, true);
                attr.put("name", agn.getGenes()[i].getName());
                attr.put("probsetname", agn.getGenes()[i].getProbsetname());
                attr.put("description", agn.getGenes()[i].getDescription());
                attr.put("index", agn.getGenes()[i].getIndex());
                node.setAttributes(attr);
                edges.add(node);
            }
        }

        graph.getGraphLayoutCache().insert(edges.toArray());
        //graph.getGraphLayoutCache().insert(vertex.toArray());
        //graph.getGraphLayoutCache().insert(blueedges.toArray());
        //graph.getGraphLayoutCache().insert(rededges.toArray());
        //graph.getGraphLayoutCache().insert(orangeedges.toArray());
        //graph.setBackground(Color.BLUE);
        //graph.getGraphLayoutCache().insert(edgesvertex.toArray());
        graph.clearSelection();

        String title = "Gene Network";
        if (agn.getTopology() != null) {
            title = title + " (" + agn.getTopology() + " topology)";
        }

        if (dolayout) {
            JGraphFacade facade = new JGraphFacade(graph); // Pass the facade the JGraph instance
            facade.setIgnoresUnconnectedCells(true);
            facade.setDirected(true);
            Rectangle2D area = new Rectangle2D.Float(0, 0, GraphDR.w, GraphDR.h);
            //JGraphLayout layout = new JGraphSelfOrganizingOrganicLayout();
            //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_TILT,GraphDR.w-15,GraphDR.h-100);
            //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_RANDOM,GraphDR.w-15,GraphDR.h-100);
            //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_CIRCLE,GraphDR.w-15,GraphDR.h-150);
            //JGraphLayout layout = new JGraphRadialTreeLayout(); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphFastOrganicLayout(); // Create an instance of the appropriate layout
            JGraphLayout layout = new JGraphOrganicLayout(area); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphTreeLayout(); // Create an instance of the appropriate layout
            layout.run(facade); // Run the layout on the facade.
            facade.scale(area);
            Map nested = facade.createNestedMap(true, true); // Obtain a map of the resulting attribute changes from the facade
            graph.getGraphLayoutCache().edit(nested); // Apply the results to the actual graph
            RefreshGraphPositionsonAGN(agn, graph);
        }
        GraphDR frame = new GraphDR(title, graph, agn);
        frame.setVisible(true);
        return (frame);
    }
    
    public static JGraph ViewAGNCLI(AGN agn) {
        // Construct Model and GraphDR
        GraphModel model = new DefaultGraphModel();
        GraphLayoutCache view = new GraphLayoutCache(model, new DefaultCellViewFactory());
        JGraph graph = new JGraph(model, view);
        // Control-drag should clone selection
        graph.setCloneable(true);
        // Enable edit without final RETURN keystroke
        graph.setInvokesStopCellEditing(true);
        // When over a cell, jump to its default port (we only
        // have one, anyway)
        graph.setJumpToDefaultPort(true);
        // Insert all three cells in one call, so we need an
        // array to store them
        List<DefaultGraphCell> vertex = new ArrayList<DefaultGraphCell>();

        for (int gt = 0; gt < agn.getGenes().length; gt++) {
            String name = String.valueOf(gt);
            if (agn.getGenes()[gt].getName() != null) {
                name = agn.getGenes()[gt].getName() + "(" + String.valueOf(gt) + ")";
            }
            float x = agn.getGenes()[gt].getX();
            float y = agn.getGenes()[gt].getY();
            DefaultGraphCell node = GraphDR.createVertex(name, x, y, 0, 0, Color.WHITE, false, true);
            AttributeMap attr = node.getAttributes();
            attr.put("index", gt);
            node.setAttributes(attr);
            vertex.add(node);
            GraphConstants.setGradientColor(vertex.get(gt).getAttributes(), Color.YELLOW);
        }
        int countedges = 0;
        for (int gt = 0; gt < agn.getGenes().length; gt++) {
            if (agn.getGenes()[gt].getPredictorsties() == null) {
                for (int gp = 0; gp < agn.getGenes()[gt].getPredictors().size(); gp++) {
                    int predictor = (Integer) agn.getGenes()[gt].getPredictors().get(gp);
                    DefaultEdge edge;
                    countedges++;
                    if (agn.getGenes()[gt].getCfvalues().size() > gp) {
                        edge = new DefaultEdge(agn.getGenes()[gt].getCfvalues().get(gp));
                    } else {
                        edge = new DefaultEdge();
                    }
                    edge.setSource(vertex.get(predictor).getChildAt(0));
                    edge.setTarget(vertex.get(gt).getChildAt(0));
                    GraphConstants.setLineEnd(edge.getAttributes(), GraphConstants.ARROW_SIMPLE);
                    GraphConstants.setRouting(edge.getAttributes(), GraphConstants.ROUTING_DEFAULT);
                    GraphConstants.setLineStyle(edge.getAttributes(), GraphConstants.STYLE_SPLINE);
                    GraphConstants.setLineWidth(edge.getAttributes(), 2);
                    GraphConstants.setLineColor(edge.getAttributes(), Color.BLACK);
                    vertex.add(edge);
                }
            } else {
                for (int tie = 0; tie < agn.getGenes()[gt].getPredictorsties().length; tie++) {
                    Vector predictorstied = agn.getGenes()[gt].getPredictorsties()[tie];
                    for (int gp = 0; gp < predictorstied.size(); gp++) {
                        int predictor = (Integer) predictorstied.get(gp);
                        DefaultEdge edge;
                        countedges++;
                        if (agn.getGenes()[gt].getCfvalues().size() > gp) {
                            edge = new DefaultEdge(agn.getGenes()[gt].getCfvalues().get(gp));
                        } else {
                            edge = new DefaultEdge();
                        }
                        edge.setSource(vertex.get(predictor).getChildAt(0));
                        edge.setTarget(vertex.get(gt).getChildAt(0));
                        GraphConstants.setLineEnd(edge.getAttributes(), GraphConstants.ARROW_SIMPLE);
                        GraphConstants.setRouting(edge.getAttributes(), GraphConstants.ROUTING_DEFAULT);
                        GraphConstants.setLineStyle(edge.getAttributes(), GraphConstants.STYLE_SPLINE);
                        GraphConstants.setLineWidth(edge.getAttributes(), 2);
                        GraphConstants.setLineColor(edge.getAttributes(), Color.BLACK);
                        vertex.add(edge);
                    }
                }
            }
        }

        graph.getGraphLayoutCache().insert(vertex.toArray());
        graph.clearSelection();

        String title = "Gene Network";
        if (agn.getTopology() != null) {
            title = title + " (" + agn.getTopology() + " topology)";
        }
        //layout do grafo, apenas se a rede for != da geografica
        if ((agn.getTopology() == null || !agn.getTopology().equalsIgnoreCase("GG")) && countedges > 0) {
            //Object roots = getRoots(); // replace getRoots with your own
            //Object array of the cell tree roots. NOTE: these are the root cell(s) of the tree(s), not the roots of the graph model.
            JGraphFacade facade = new JGraphFacade(graph); // Pass the facade the JGraph instance
            facade.setIgnoresUnconnectedCells(true);
            facade.setDirected(true);
            Rectangle2D area = new Rectangle2D.Float(15, 15, GraphDR.w - 15, GraphDR.h - 100);
            //JGraphLayout layout = new JGraphSelfOrganizingOrganicLayout();
            //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_TILT,GraphDR.w-15,GraphDR.h-100);
            //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_RANDOM,GraphDR.w-15,GraphDR.h-100);
            JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_CIRCLE,GraphDR.w-15,GraphDR.h-150);
            //JGraphLayout layout = new JGraphRadialTreeLayout(); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphFastOrganicLayout(); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphOrganicLayout(area); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphTreeLayout(); // Create an instance of the appropriate layout

            layout.run(facade); // Run the layout on the facade.
            facade.scale(area);
            Map nested = facade.createNestedMap(true, true); // Obtain a map of the resulting attribute changes from the facade
            graph.getGraphLayoutCache().edit(nested); // Apply the results to the actual graph
        }
        if (countedges > 0){
            GraphDR frame = new GraphDR(title, graph, agn);
            frame.setVisible(true);
        }else{
            // JOptionPane.showMessageDialog(null, "The method found no relationship to the selected targets.", "Information", JOptionPane.INFORMATION_MESSAGE);
            IOFile.PrintlnAndLog("The method found no relationship to the selected targets.");
        }
        return (graph);
    }

    /**
     * Creates an adjacency matrix representation of the gene network structure.
     * <p>
     * This method transforms the network structure stored in an AGN object into
     * a binary adjacency matrix representation, where each element [i,j] indicates
     * whether gene i regulates gene j. A value of 1 indicates that there is a 
     * directed regulatory connection from gene i to gene j, while a value of 0
     * indicates no direct regulation.
     * </p>
     * 
     * <p>
     * The method handles both standard predictors and predictors organized in 
     * "ties" (groups of predictors that collectively regulate a target). All 
     * regulatory relationships are represented in the resulting matrix, making
     * it suitable for further analysis, visualization, or export to other tools.
     * </p>
     * 
     * <p>
     * The adjacency matrix is a common representation of network structure that
     * can be easily saved to disk, used for graph analysis, or visualized using
     * various tools. It's particularly useful for comparing inferred networks
     * with known regulatory relationships or for calculating network statistics.
     * </p>
     *
     * @param agn The artificial gene network object containing the network structure.
     * @return A square binary adjacency matrix where a 1 at position [i,j] indicates
     *         that gene i regulates gene j.
     * @see MainCLI#networkInferenceActionPerformed()
     * @see IOFile#WriteMatrix(String, int[][], String)
     */
    public static int[][] createAdjacencyMatrix(AGN agn) {
            int numGenes = agn.getGenes().length;
            int[][] adjacencyMatrix = new int[numGenes][numGenes];

            for (int gt = 0; gt < numGenes; gt++) {
                if (agn.getGenes()[gt].getPredictorsties() == null) {
                    for (int gp = 0; gp < agn.getGenes()[gt].getPredictors().size(); gp++) {
                        int predictor = (Integer) agn.getGenes()[gt].getPredictors().get(gp);
                        adjacencyMatrix[predictor][gt] = 1; // or use the weight if applicable
                    }
                } else {
                    for (int tie = 0; tie < agn.getGenes()[gt].getPredictorsties().length; tie++) {
                        Vector predictorstied = agn.getGenes()[gt].getPredictorsties()[tie];
                        for (int gp = 0; gp < predictorstied.size(); gp++) {
                            int predictor = (Integer) predictorstied.get(gp);
                            adjacencyMatrix[predictor][gt] = 1; // or use the weight if applicable
                        }
                    }
                }
            }

            return adjacencyMatrix;
        }

    public static JGraph ViewAGN(AGN agn) {
        // Construct Model and GraphDR
        GraphModel model = new DefaultGraphModel();
        GraphLayoutCache view = new GraphLayoutCache(model, new DefaultCellViewFactory());
        JGraph graph = new JGraph(model, view);
        // Control-drag should clone selection
        graph.setCloneable(true);
        // Enable edit without final RETURN keystroke
        graph.setInvokesStopCellEditing(true);
        // When over a cell, jump to its default port (we only
        // have one, anyway)
        graph.setJumpToDefaultPort(true);
        // Insert all three cells in one call, so we need an
        // array to store them
        List<DefaultGraphCell> vertex = new ArrayList<DefaultGraphCell>();

        for (int gt = 0; gt < agn.getGenes().length; gt++) {
            String name = String.valueOf(gt);
            if (agn.getGenes()[gt].getName() != null) {
                name = agn.getGenes()[gt].getName() + "(" + String.valueOf(gt) + ")";
            }
            float x = agn.getGenes()[gt].getX();
            float y = agn.getGenes()[gt].getY();
            DefaultGraphCell node = GraphDR.createVertex(name, x, y, 0, 0, Color.WHITE, false, true);
            AttributeMap attr = node.getAttributes();
            attr.put("index", gt);
            node.setAttributes(attr);
            vertex.add(node);
            GraphConstants.setGradientColor(vertex.get(gt).getAttributes(), Color.YELLOW);
        }
        int countedges = 0;
        for (int gt = 0; gt < agn.getGenes().length; gt++) {
            if (agn.getGenes()[gt].getPredictorsties() == null) {
                for (int gp = 0; gp < agn.getGenes()[gt].getPredictors().size(); gp++) {
                    int predictor = (Integer) agn.getGenes()[gt].getPredictors().get(gp);
                    DefaultEdge edge;
                    countedges++;
                    if (agn.getGenes()[gt].getCfvalues().size() > gp) {
                        edge = new DefaultEdge(agn.getGenes()[gt].getCfvalues().get(gp));
                    } else {
                        edge = new DefaultEdge();
                    }
                    edge.setSource(vertex.get(predictor).getChildAt(0));
                    edge.setTarget(vertex.get(gt).getChildAt(0));
                    GraphConstants.setLineEnd(edge.getAttributes(), GraphConstants.ARROW_SIMPLE);
                    GraphConstants.setRouting(edge.getAttributes(), GraphConstants.ROUTING_DEFAULT);
                    GraphConstants.setLineStyle(edge.getAttributes(), GraphConstants.STYLE_SPLINE);
                    GraphConstants.setLineWidth(edge.getAttributes(), 2);
                    GraphConstants.setLineColor(edge.getAttributes(), Color.BLACK);
                    vertex.add(edge);
                }
            } else {
                for (int tie = 0; tie < agn.getGenes()[gt].getPredictorsties().length; tie++) {
                    Vector predictorstied = agn.getGenes()[gt].getPredictorsties()[tie];
                    for (int gp = 0; gp < predictorstied.size(); gp++) {
                        int predictor = (Integer) predictorstied.get(gp);
                        DefaultEdge edge;
                        countedges++;
                        if (agn.getGenes()[gt].getCfvalues().size() > gp) {
                            edge = new DefaultEdge(agn.getGenes()[gt].getCfvalues().get(gp));
                        } else {
                            edge = new DefaultEdge();
                        }
                        edge.setSource(vertex.get(predictor).getChildAt(0));
                        edge.setTarget(vertex.get(gt).getChildAt(0));
                        GraphConstants.setLineEnd(edge.getAttributes(), GraphConstants.ARROW_SIMPLE);
                        GraphConstants.setRouting(edge.getAttributes(), GraphConstants.ROUTING_DEFAULT);
                        GraphConstants.setLineStyle(edge.getAttributes(), GraphConstants.STYLE_SPLINE);
                        GraphConstants.setLineWidth(edge.getAttributes(), 2);
                        GraphConstants.setLineColor(edge.getAttributes(), Color.BLACK);
                        vertex.add(edge);
                    }
                }
            }
        }

        graph.getGraphLayoutCache().insert(vertex.toArray());
        graph.clearSelection();

        String title = "Gene Network";
        if (agn.getTopology() != null) {
            title = title + " (" + agn.getTopology() + " topology)";
        }
        //layout do grafo, apenas se a rede for != da geografica
        if ((agn.getTopology() == null || !agn.getTopology().equalsIgnoreCase("GG")) && countedges > 0) {
            //Object roots = getRoots(); // replace getRoots with your own
            //Object array of the cell tree roots. NOTE: these are the root cell(s) of the tree(s), not the roots of the graph model.
            JGraphFacade facade = new JGraphFacade(graph); // Pass the facade the JGraph instance
            facade.setIgnoresUnconnectedCells(true);
            facade.setDirected(true);
            Rectangle2D area = new Rectangle2D.Float(15, 15, GraphDR.w - 15, GraphDR.h - 100);
            //JGraphLayout layout = new JGraphSelfOrganizingOrganicLayout();
            //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_TILT,GraphDR.w-15,GraphDR.h-100);
            //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_RANDOM,GraphDR.w-15,GraphDR.h-100);
            JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_CIRCLE,GraphDR.w-15,GraphDR.h-150);
            //JGraphLayout layout = new JGraphRadialTreeLayout(); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphFastOrganicLayout(); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphOrganicLayout(area); // Create an instance of the appropriate layout
            //JGraphLayout layout = new JGraphTreeLayout(); // Create an instance of the appropriate layout

            layout.run(facade); // Run the layout on the facade.
            facade.scale(area);
            Map nested = facade.createNestedMap(true, true); // Obtain a map of the resulting attribute changes from the facade
            graph.getGraphLayoutCache().edit(nested); // Apply the results to the actual graph
        }
        if (countedges > 0){
            GraphDR frame = new GraphDR(title, graph, agn);
            frame.setVisible(true);
        }else{
            JOptionPane.showMessageDialog(null, "The method found no relationship to the selected targets.", "Information", JOptionPane.INFORMATION_MESSAGE);
        }
        return (graph);
    }

    public static float[][] ConcatenateSignalSeparating(float[][] m1, float[][] m2) {
        int lines = m1.length;
        int columns1 = m1[0].length;
        int columns2 = m2[0].length;

        float[][] nm = new float[lines][columns1 + columns2 + 1];

        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < columns1; j++) {
                nm[i][j] = m1[i][j];
            }

            nm[i][columns1] = -999;

            for (int j = columns1; j < (columns1 + columns2); j++) {
                nm[i][j + 1] = m2[i][j - columns1];
            }
        }
        return (nm);
    }

    public static int[][] ConcatenateSignalSeparatingq(int[][] m1, int[][] m2) {
        int lines = m1.length;
        int columns1 = m1[0].length;
        int columns2 = m2[0].length;

        int[][] nm = new int[lines][columns1 + columns2 + 1];

        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < columns1; j++) {
                nm[i][j] = m1[i][j];
            }

            nm[i][columns1] = -999;

            for (int j = columns1; j < (columns1 + columns2); j++) {
                nm[i][j + 1] = m2[i][j - columns1];
            }
        }
        return (nm);
    }

    public static float[][] ConcatenateSignal(float[][] m1, float[][] m2) {
        int lines = m1.length;
        int columns1 = m1[0].length;
        int columns2 = m2[0].length;

        float[][] nm = new float[lines][columns1 + columns2];

        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < columns1; j++) {
                nm[i][j] = m1[i][j];
            }
            for (int j = columns1; j < (columns1 + columns2); j++) {
                nm[i][j] = m2[i][j - columns1];
            }
        }
        return (nm);
    }

    public static int[][] ConcatenateSignalq(int[][] m1, int[][] m2) {
        int lines = m1.length;
        int columns1 = m1[0].length;
        int columns2 = m2[0].length;

        int[][] nm = new int[lines][columns1 + columns2];

        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < columns1; j++) {
                nm[i][j] = m1[i][j];
            }
            for (int j = columns1; j < (columns1 + columns2); j++) {
                nm[i][j] = m2[i][j - columns1];
            }
        }
        return (nm);
    }

    public static void setNameandType(AGN agn, Vector[] nameandtype) {
        for (int g = 0; g < agn.getGenes().length; g++) {
            String name = (String) nameandtype[0].get(g);
            String type = (String) nameandtype[1].get(g);
            agn.getGenes()[g].setName(name);
            agn.getGenes()[g].setType(type);
            //usando o gene new
            //agn.getGenes()[g].setClasse(Integer.valueOf(type));
        }
    }

    public static void setGeneNames(AGN agn, Vector names) {
        if (agn.getNrgenes() == names.size()) {
            for (int g = 0; g < agn.getGenes().length; g++) {
                String name = (String) names.get(g);
                agn.getGenes()[g].setName(name);
            }
        } else {
            IOFile.PrintlnAndLog("Error on labeling genes, size does not match.");
        }
    }

    public static void setPSNandClass(AGN agn, Vector[] nameandtype) {
        for (int g = 0; g < agn.getGenes().length; g++) {
            String psn = (String) nameandtype[0].get(g);
            int classe = Integer.valueOf(((String) nameandtype[1].get(g)));
            agn.getGenes()[g].setProbsetname(psn);
            agn.getGenes()[g].setClasse(classe);
        }
    }

    /**
     * Creates a training set for inferring temporal relationships for a target gene.
     * <p>
     * This method rearranges the gene expression data matrix to prepare it for feature
     * selection algorithms that will identify regulatory relationships. It restructures
     * the data to allow for time-delayed analysis, where the expression of predictor
     * genes at time t is used to predict the expression of the target gene at time t+1.
     * </p>
     * 
     * <p>
     * The method performs the following transformations on the data:
     * <ol>
     *   <li>Removes the target gene from the matrix of predictors</li>
     *   <li>Creates a new column containing the target gene's expression values</li>
     *   <li>Shifts the target gene values to represent the t+1 relationship</li>
     *   <li>For periodic signals, connects the last timepoint back to the first</li>
     *   <li>Removes rows with invalid values (marked with skipvalue)</li>
     * </ol>
     * </p>
     * 
     * <p>
     * This transformation is essential for the feature selection algorithms to correctly
     * identify time-delayed regulatory relationships in temporal gene expression data.
     * </p>
     *
     * @param agn The AGN object containing the quantized temporal expression data
     * @param target The index of the target gene to analyze
     * @param isPeriodic Whether the time series should be treated as periodic (circular)
     * @return A character matrix representing the training set for feature selection
     * @see #processTargetForNetworkRecovery(AGN, AGN, int, boolean, float, String, float, float, float, int, int, boolean, int, String, int)
     */
    public static char[][] MakeTemporalTrainingSet(AGN agn, int target, boolean isPeriodic) {
        int rowsoriginal = agn.getTemporalsignalquantized().length;
        int colsoriginal = agn.getTemporalsignalquantized()[0].length;
        int rowsts = colsoriginal;

        //verificacao se o sinal eh periodico.
        if (!isPeriodic) {
            rowsts--;//desconsidera a observacao da ultima coluna tentando predizer a primeira.
        }

        char[][] trainingset = new char[rowsts][rowsoriginal];

        //fill the columns before target row
        for (int row = 0; row < target; row++) {
            for (int col = 0; col < rowsts; col++) {
                int temp = (int) agn.getTemporalsignalquantized()[row][col];
                trainingset[col][row] = (char) temp;
            }
        }

        //fill the columns after target row
        for (int row = target + 1; row < rowsoriginal; row++) {
            for (int col = 0; col < rowsts; col++) {
                int temp = (int) agn.getTemporalsignalquantized()[row][col];
                trainingset[col][row - 1] = (char) temp;
            }
        }

        //fill the target row in the last column of the training set.
        for (int col = 1; col <= rowsts; col++) {
            int temp = (int) agn.getTemporalsignalquantized()[target][col % colsoriginal];
            trainingset[col - 1][rowsoriginal - 1] = (char) temp;
        }

        //verificar quais linhas possuem caracter -999 e exclui-las da matriz.
        Vector rowsfr = new Vector();
        for (int i = 0; i < trainingset.length; i++) {
            boolean remove = false;
            for (int j = 0; j < trainingset[0].length && !remove; j++) {
                if (trainingset[i][j] == (char) Preprocessing.skipvalue) {
                    remove = true;
                }
            }
            if (remove) {
                rowsfr.add(i);
            }
        }

        if (rowsfr.size() > 0) {
            char[][] newtrainingset = new char[trainingset.length - rowsfr.size()][trainingset[0].length];
            int newrow = 0;
            for (int i = 0; i < trainingset.length; i++) {
                if (rowsfr.contains(i)) {
                    newrow++;
                    continue;
                }
                for (int j = 0; j < trainingset[0].length; j++) {
                    newtrainingset[i - newrow][j] = trainingset[i][j];
                }
            }
            trainingset = newtrainingset;
        }
        return (trainingset);
    }

    /**
     * Creates a training set for inferring steady-state relationships for a target gene.
     * <p>
     * This method rearranges the gene expression data matrix to prepare it for feature
     * selection algorithms that will identify regulatory relationships in steady-state data.
     * Unlike the temporal training set, this method doesn't consider time-delayed effects,
     * but rather looks for relationships within the same experimental conditions or samples.
     * </p>
     * 
     * <p>
     * The method performs the following transformations on the data:
     * <ol>
     *   <li>Removes the target gene from the matrix of predictors</li>
     *   <li>Creates a new column containing the target gene's expression values</li>
     *   <li>For each sample, uses all other genes' values to predict the target gene's value</li>
     *   <li>Removes rows with invalid values (marked with skipvalue)</li>
     * </ol>
     * </p>
     * 
     * <p>
     * This transformation allows the feature selection algorithms to identify which genes
     * have expression patterns that are predictive of the target gene's expression across
     * different experimental conditions, suggesting potential regulatory relationships.
     * </p>
     *
     * @param agn The AGN object containing the quantized expression data
     * @param target The index of the target gene to analyze
     * @return A character matrix representing the training set for feature selection
     * @see #processTargetForNetworkRecovery(AGN, AGN, int, boolean, float, String, float, float, float, int, int, boolean, int, String, int)
     */
    public static char[][] MakeSteadyStateTrainingSet(AGN agn, int target) {
        int rowsoriginal = agn.getTemporalsignalquantized().length;
        int colsoriginal = agn.getTemporalsignalquantized()[0].length;
        int rowsts = colsoriginal;
        char[][] trainingset = new char[rowsts][rowsoriginal];
        //fill the columns before target row
        for (int row = 0; row < target; row++) {
            for (int col = 0; col < rowsts; col++) {
                int temp = (int) agn.getTemporalsignalquantized()[row][col];
                trainingset[col][row] = (char) temp;
            }
        }
        //fill the columns after target row
        for (int row = target + 1; row < rowsoriginal; row++) {
            for (int col = 0; col < rowsts; col++) {
                int temp = (int) agn.getTemporalsignalquantized()[row][col];
                trainingset[col][row - 1] = (char) temp;
            }
        }
        //fill the target row in the last column of the training set.
        for (int col = 0; col < rowsts; col++) {
            int temp = (int) agn.getTemporalsignalquantized()[target][col];
            trainingset[col][rowsoriginal - 1] = (char) temp;
        }

        //verificar quais linhas possuem caracter -999 e exclui-las da matriz.
        Vector rowsfr = new Vector();
        for (int i = 0; i < trainingset.length; i++) {
            boolean remove = false;
            for (int j = 0; j < trainingset[0].length && !remove; j++) {
                if (trainingset[i][j] == (char) Preprocessing.skipvalue) {
                    remove = true;
                }
            }
            if (remove) {
                rowsfr.add(i);
            }
        }

        if (rowsfr.size() > 0) {
            char[][] newtrainingset = new char[trainingset.length - rowsfr.size()][trainingset[0].length];
            int newrow = 0;
            for (int i = 0; i < trainingset.length; i++) {
                if (rowsfr.contains(i)) {
                    newrow++;
                    continue;
                }
                for (int j = 0; j < trainingset[0].length; j++) {
                    newtrainingset[i - newrow][j] = trainingset[i][j];
                }
            }
            trainingset = newtrainingset;
        }
        return (trainingset);
    }

    public static void CreateTemporalSignalq(AGN agn) {
        //cria a matriz para armazenar os dados temporais
        float[][] temporalsignal = new float[agn.getNrgenes()][agn.getSignalsize()];
        //inicializa o primeiro instante de tempo com os valores atuais dos genes.
        for (int i = 0; i < agn.getNrgenes(); i++) {
            temporalsignal[i][0] = agn.getGenes()[i].getValue();
        }
        //cria o sinal, observando os preditores e as funcoes booleanas associadas a ele.
        for (int time = 1; time < agn.getSignalsize(); time++) {
            for (int target = 0; target < agn.getNrgenes(); target++) {
                //gera o valor do gene target no tempo time, observando seus preditores
                //e funcoes booleanas associadas a ele no instante de tempo time-1.
                boolean genevalue = Simulation.ApplyLogicalCircuit(agn, target, time - 1, temporalsignal);
                int gv = MathRoutines.Boolean2Int(genevalue);
                temporalsignal[target][time] = gv;
            }
        }
        agn.setTemporalsignalquantized(temporalsignal);
    }

    public static void CreateSignalInitializations(AGN agn, boolean separatesignal) {
        //armazena os valores iniciais dos genes da rede.
        float[] initialvalues = agn.getInitialValues();
        //cria o sinal temporal a partir dos valores iniciais dos genes e
        //aplicacao das funcoes e preditores associados a eles.
        //AGNRoutines.CreateTemporalSignal(agn);

        //gera o sinal de expressao simulado.
        int[][] generated_data = agn.getTemporalsignalquantized();

        //inicializa a geracao de pseudo-aleatorios utilizando a hora do sistema como semente.
        Random rn = new Random(System.nanoTime());

        //gera novas inicializacoes (concatenacoes de sinal).
        for (int concat = 1; concat <= agn.getNrinitializations(); concat++) {
            float[] newinitialvalues = new float[agn.getNrgenes()];
            for (int i = 0; i < agn.getNrgenes(); i++) {
                newinitialvalues[i] = rn.nextInt(agn.getQuantization());
            }
            agn.setInitialValues(newinitialvalues);
            AGNRoutines.CreateTemporalSignalq(agn);
            int[][] otherinicializationdata = agn.getTemporalsignalquantized();
            //concatena os dados gerados pelo estado inicial anterior e o novo estado inicial.
            if (separatesignal) {
                generated_data = AGNRoutines.ConcatenateSignalSeparatingq(generated_data, otherinicializationdata);
            } else {
                generated_data = AGNRoutines.ConcatenateSignalq(generated_data, otherinicializationdata);
            }
        }
        //retorna os valores iniciais originais aos genes.
        agn.setInitialValues(initialvalues);
        agn.setTemporalsignalquantized(generated_data);
    }

    // Helper class to hold per-target results
    /**
     * A container class that holds the results of network inference for a single target gene.
     * <p>
     * This class encapsulates all the data related to the recovery of predictors (regulators)
     * for a specific target gene during the network inference process. It stores the identified
     * predictors, ties between different predictor sets, probability tables, criterion function
     * values, and text descriptions of the results.
     * </p>
     * <p>
     * This result object is used for intermediate storage during the parallel processing
     * of multiple target genes in the network inference process.
     * </p>
     */
    private static class TargetRecoveryResult {
        public Vector predictors;
        public Vector ties;
        public Vector[] predictorsties;
        public Vector probtable;
        public float hGlobal;
        public StringBuffer txt;
        public int targetindex;
        public TargetRecoveryResult(int targetindex) {
            this.targetindex = targetindex;
            this.predictors = new Vector();
            this.ties = new Vector();
            this.predictorsties = null;
            this.probtable = null;
            this.hGlobal = 1.0f;
            this.txt = new StringBuffer();
        }
    }

    /**
     * Processes a single target gene to discover its regulatory connections.
     * <p>
     * This method is the core of the network inference algorithm, processing one target gene at a time.
     * It performs feature selection to identify which genes are likely regulators of the target gene,
     * based on information theory metrics like entropy and mutual information.
     * </p>
     * 
     * <p>
     * The method performs the following steps:
     * <ol>
     *   <li>Creates a training set representing either temporal or steady-state relationships</li>
     *   <li>Applies feature selection algorithm (SFS, SFFS, or Exhaustive Search) to identify predictors</li>
     *   <li>Filters results based on an entropy threshold</li>
     *   <li>Handles ties between different predictor sets</li>
     *   <li>Records results and generates detailed output</li>
     * </ol>
     * </p>
     * 
     * <p>
     * This method is designed to be called in parallel across multiple target genes, with
     * results later aggregated to form the complete gene regulatory network.
     * </p>
     *
     * @param recoveredagn The AGN object where the reconstructed network will be stored
     * @param originalagn The original AGN object for comparison (can be null)
     * @param datatype The type of data: 1 for time-series, 2 for steady-state
     * @param isPeriodic Whether the time-series data is periodic
     * @param threshold_entropy Threshold value for determining edges in the inferred network
     * @param type_entropy The penalization method: "no_obs" or "poor_obs"
     * @param alpha Alpha value for no_obs penalization method
     * @param beta Beta value for poor_obs penalization method
     * @param q_entropy Q-entropy parameter (1 for Shannon entropy, other values for Tsallis entropy)
     * @param maxfeatures Maximum number of features/predictors to consider
     * @param searchalgorithm The feature selection algorithm to use: 1=SFS, 2=Exhaustive, 3=SFFS, 4=SFFS_stack
     * @param targetaspredictors Whether to invert the relationship and use targets as predictors
     * @param resultsetsize Number of top feature sets to keep in results
     * @param tiesout Path to output file for storing tie information (can be null)
     * @param targetindex Index of the target gene being processed
     * @return A TargetRecoveryResult containing all discovered regulatory relationships for this target
     * @see fs.FS For the feature selection algorithms used
     * @see #MakeTemporalTrainingSet(AGN, int, boolean)
     * @see #MakeSteadyStateTrainingSet(AGN, int)
     */
    private static TargetRecoveryResult processTargetForNetworkRecovery(
            AGN recoveredagn, AGN originalagn, int datatype, boolean isPeriodic, float threshold_entropy,
            String type_entropy, float alpha, float beta, float q_entropy, int maxfeatures, int searchalgorithm,
            boolean targetaspredictors, int resultsetsize, String tiesout, int targetindex) {
        TargetRecoveryResult result = new TargetRecoveryResult(targetindex);
        char[][] strainingset;
        if (datatype == 1) {
            strainingset = MakeTemporalTrainingSet(recoveredagn, targetindex, isPeriodic);
        } else {
            strainingset = MakeSteadyStateTrainingSet(recoveredagn, targetindex);
        }
        FS fs = new FS(strainingset, recoveredagn.getQuantization(),
                recoveredagn.getQuantization(),
                type_entropy, alpha, beta, q_entropy, resultsetsize);

        if (CNMeasurements.hasVariation(strainingset, isPeriodic)) {
            timer.start("running_search_algorithm-target_index_" + targetindex);
            //IOFile.PrintMatrix(strainingset);
            //IOFile.PrintlnAndLog("\nTarget = " + gt);
            /*
            n: number of possible values for features
            c: number of possible classes
            NO CASO DA GERACAO DAS REDES, ASSUMIMOS QUE N=C=2.
             */
            if (searchalgorithm == 1) {
                fs.runSFS(false, maxfeatures);
            } else if (searchalgorithm == 3) {
                fs.runSFFS(maxfeatures, targetindex, recoveredagn);
            } else if (searchalgorithm == 4) {
                //implementacao do SFFS usando pilha de execucao para expandir todos os
                //empates identificados.
                fs.runSFFS_stack(maxfeatures, targetindex, recoveredagn);
            } else if (searchalgorithm == 2) {
                fs.runSFS(true, maxfeatures); /* a call to SFS is made in order to get the
                //ideal dimension to run the exhaustive search;*/
                int itmax = fs.itmax;
                FS fsPrev = new FS(strainingset, recoveredagn.getQuantization(), recoveredagn.getQuantization(), type_entropy, alpha, beta,
                        q_entropy, resultsetsize);
                for (int i = 1; i <= itmax; i++) {
                    fs = new FS(strainingset, recoveredagn.getQuantization(), recoveredagn.getQuantization(), type_entropy, alpha, beta,
                            q_entropy, resultsetsize);
                    fs.itmax = i;
                    fs.runExhaustive(0, 0, fs.I);
                    if (fs.hGlobal < fsPrev.hGlobal) {
                        fsPrev = fs;
                    } else {
                        fs = fsPrev;
                        break;
                    }
                }
            }
            timer.end("running_search_algorithm-target_index_" + targetindex);

            if (targetaspredictors) {
                result.txt.append("Predictor: " + (targetindex) + " name:" + recoveredagn.getGenes()[targetindex].getName() + "\nTargets: ");
                IOFile.PrintAndLog("Predictor: " + (targetindex) + " name:" + recoveredagn.getGenes()[targetindex].getName() + "\nTargets: ");
            } else {
                result.txt.append("Target: " + (targetindex) + " name:" + recoveredagn.getGenes()[targetindex].getName() + "\nPredictors: ");
                IOFile.PrintAndLog("\nTarget: " + (targetindex) + " name:" + recoveredagn.getGenes()[targetindex].getName() + "\nPredictors: ");
            }

            for (int i = 0; i < fs.I.size(); i++) {
                int predictor_gene = Integer.valueOf(fs.I.elementAt(i).toString());
                if (predictor_gene >= targetindex) {
                    predictor_gene++;
                }
                if (fs.hGlobal < threshold_entropy) {
                    //int predictor_gene = Integer.valueOf(fs.I.elementAt(i).toString());
                    //if (geneid != null)
                    //    predictor_gene = (Integer) geneid[predictor_gene].get(1);
                    result.txt.append(predictor_gene + " name:" + recoveredagn.getGenes()[predictor_gene].getName() + " ");
                    IOFile.PrintAndLog(predictor_gene + " name:" + recoveredagn.getGenes()[predictor_gene].getName() + " ");
                    result.predictors.add(predictor_gene);
                } else {
                    if (targetaspredictors) {
                        result.txt.append("\ntarget " + predictor_gene + " excluded by threshold. Criterion Function Value = " + fs.hGlobal);
                        IOFile.PrintAndLog("\ntarget " + predictor_gene + " excluded by threshold. Criterion Function Value = " + fs.hGlobal);
                    } else {
                        result.txt.append("\npredictor " + predictor_gene + " excluded by threshold. Criterion Function Value = " + fs.hGlobal);
                        IOFile.PrintAndLog("\npredictor " + predictor_gene + " excluded by threshold. Criterion Function Value = " + fs.hGlobal);
                    }
                }
            }

            int s = fs.I.size();
            if ((searchalgorithm == 3 || searchalgorithm == 4) && fs.ties[s] != null && fs.ties[s].size() > 1 && fs.hGlobal < threshold_entropy) {
                //detectou empate entre grupos de preditores.
                result.txt.append("\nPredictors Ties: ");
                IOFile.PrintAndLog("\nPredictors Ties: ");

                //vetor para armazenar os preditores que empataram ao predizer o target.
                //cada posicao do vetor, tambem eh um vetor contendo um conjunto de preditores (indices inteiros).
                Vector[] predictorsties = new Vector[fs.ties[s].size()];

                for (int j = 0; j < fs.ties[s].size(); j++) {

                    //inicializacao de cada posicao do vetor com um novo vetor.
                    predictorsties[j] = new Vector();

                    //armazena os valores empatados apenas com entropia
                    //igual a melhor resposta.
                    Vector item = (Vector) fs.ties[s].get(j);
                    Vector tie = new Vector();
                    for (int k = 0; k < item.size(); k++) {
                        int geneindex = (Integer) item.get(k);
                        if (geneindex >= targetindex) {
                            geneindex++;
                        }

                        //adiciona o indice do preditor empatado ao conjunto.
                        predictorsties[j].add(k, geneindex);

                        result.txt.append(geneindex + " name:" + recoveredagn.getGenes()[geneindex].getName() + " ");
                        IOFile.PrintAndLog(geneindex + " name:" + recoveredagn.getGenes()[geneindex].getName() + " ");
                        tie.add(geneindex);
                    }
                    IOFile.PrintAndLog(" (" + fs.jointentropiesties[j] + ") ");
                    result.txt.append("\t");
                    IOFile.PrintAndLog("\t");
                    result.ties.add(tie);
                }
                result.predictorsties = predictorsties;
            }
            result.probtable = fs.probtable;
            result.hGlobal = fs.hGlobal;
        } else {
            if (targetaspredictors) {
                IOFile.PrintAndLog("Predictor " + targetindex + " name " + recoveredagn.getGenes()[targetindex].getName() + ", has no variation on its values.");
                result.txt.append("Predictor " + targetindex + " name " + recoveredagn.getGenes()[targetindex].getName() + ", has no variation on its values.");
            } else {
                IOFile.PrintAndLog("Target " + targetindex + " name " + recoveredagn.getGenes()[targetindex].getName() + ", has no variation on its values.");
                result.txt.append("Target " + targetindex + " name " + recoveredagn.getGenes()[targetindex].getName() + ", has no variation on its values.");
            }
            result.hGlobal = fs.hGlobal;
        }
        IOFile.PrintlnAndLog("\nCriterion Function Value: " + fs.hGlobal);
        result.txt.append("\nCriterion Function Value: " + fs.hGlobal + "\n");
        IOFile.PrintAndLog("\n");
        result.txt.append("\n");
        return result;
    }

    /**
     * Recovers a gene regulatory network from temporal or steady-state expression data.
     * <p>
     * This is the core algorithm that infers gene regulatory relationships from expression data.
     * The method analyzes each target gene separately, using feature selection algorithms to
     * identify which genes are the most likely regulators (predictors) of the target based on
     * information-theoretic measures.
     * </p>
     * 
     * <p>
     * The method supports both time-series and steady-state data analysis:
     * <ul>
     *   <li>For time-series data (datatype=1), it looks for relationships between gene expression
     *       at time t and time t+1, considering potential time-delayed regulatory effects.</li>
     *   <li>For steady-state data (datatype=2), it analyzes relationships within the same
     *       time point across different experimental conditions or samples.</li>
     * </ul>
     * </p>
     * 
     * <p>
     * The network inference process uses entropy-based feature selection (SFS, SFFS, or Exhaustive Search)
     * to identify the most informative set of predictor genes for each target gene. The method
     * can process target genes in parallel using a configurable threading strategy and supports
     * various options for penalization, entropy type (Shannon or Tsallis), and thresholding.
     * </p>
     * 
     * <p>
     * The resulting network is stored in the provided recoveredagn object, with each gene
     * containing information about its predictors and the associated confidence values.
     * </p>
     *
     * @param recoveredagn The AGN object where the reconstructed network will be stored
     * @param originalagn The original AGN object for comparison (can be null)
     * @param datatype The type of data: 1 for time-series, 2 for steady-state
     * @param isPeriodic Whether the time-series data is periodic (last timepoint connects to first)
     * @param threshold_entropy Threshold value for determining edges in the inferred network
     * @param type_entropy The penalization method: "no_obs" or "poor_obs"
     * @param alpha Alpha value for no_obs penalization method
     * @param beta Beta value for poor_obs penalization method
     * @param q_entropy Q-entropy parameter (1 for Shannon entropy, other values for Tsallis entropy)
     * @param targets Vector of target gene indexes to analyze (if null, all genes are considered)
     * @param maxfeatures Maximum number of features/predictors to consider for each target
     * @param searchalgorithm The feature selection algorithm to use: 1=SFS, 2=Exhaustive, 3=SFFS, 4=SFFS_stack
     * @param targetaspredictors Whether to invert the relationship and use targets as predictors
     * @param resultsetsize Number of top feature sets to keep in results
     * @param tiesout Path to output file for storing tie information (can be null)
     * @param threadDistribution Strategy for parallel processing: "sequential", "demand", or "spaced"
     * @param numberOfThreads Number of threads to use for parallel processing
     * @return A StringBuffer containing detailed output of the network inference process
     * @see fs.FS For the feature selection algorithms used
     * @see MainCLI#networkInferenceActionPerformed() Where this method is called
     */
    public static StringBuffer RecoverNetworkfromTemporalExpression(
            AGN recoveredagn, AGN originalagn, int datatype,
            boolean isPeriodic, float threshold_entropy,
            String type_entropy, float alpha, float beta, float q_entropy,
            Vector targets, int maxfeatures, int searchalgorithm,
            boolean targetaspredictors, int resultsetsize,
            String tiesout,
            String threadDistribution, // "sequential", "demand", "spaced"
            int numberOfThreads // pass directly from MainCLI
    ) {
        StringBuffer txt = new StringBuffer();
        int rows = recoveredagn.getTemporalsignalquantized().length;

        IOFile.PrintAndLog("\n\n");
        txt.append("\n\n");

        //Caso o vetor targets seja nulo, e assumido que todas
        //as linhas da matriz serao targets.
        if (targets == null) {
            targets = new Vector();
            for (int i = 0; i < rows; i++) {
                targets.add(String.valueOf(i));
            }
        }

        // Convert targets to List<Integer>
        java.util.List<Integer> targetIndices = new java.util.ArrayList<>();
        for (int ig = 0; ig < targets.size(); ig++) {
            targetIndices.add(Integer.valueOf((String) targets.get(ig)));
        }

        ThreadManager threadManager = new ThreadManager();
        Object lock = new Object();

        ThreadManager.TargetProcessor processor = (int targetindex) -> {
            TargetRecoveryResult result = processTargetForNetworkRecovery(
                recoveredagn, originalagn, datatype, isPeriodic, threshold_entropy, type_entropy, alpha, beta, q_entropy,
                maxfeatures, searchalgorithm, targetaspredictors, resultsetsize, tiesout, targetindex);
            synchronized (lock) {
                txt.append(result.txt);
                // Now update AGN object as before
                if (result.predictors != null && result.predictors.size() > 0) {
                    for (int i = 0; i < result.predictors.size(); i++) {
                        int predictor_gene = (Integer) result.predictors.get(i);
                        recoveredagn.getGenes()[targetindex].addPredictor(predictor_gene, result.hGlobal);
                        recoveredagn.getGenes()[predictor_gene].addTarget(targetindex);
                        if (result.probtable != null) {
                            recoveredagn.getGenes()[targetindex].setProbtable(result.probtable);
                        }
                    }
                }
                if (result.predictorsties != null) {
                    recoveredagn.getGenes()[targetindex].setPredictorsties(result.predictorsties);
                }
                // Write ties if needed
                if (tiesout != null && originalagn != null) {
                    Vector originalpredictors = originalagn.getGenes()[targetindex].getPredictors();
                    IOFile.WriteTies(
                        originalagn,
                        tiesout,
                        targetindex,
                        (int) originalagn.getAvgedges(),
                        originalagn.getTopology(),
                        originalpredictors,
                        q_entropy,
                        result.predictors,
                        result.ties,
                        result.hGlobal,
                        false);
                }
            }
        };

        if ("demand".equalsIgnoreCase(threadDistribution)) {
            threadManager.executeThreadsByDynamicAssignment(targetIndices, processor, numberOfThreads);
        } else if ("spaced".equalsIgnoreCase(threadDistribution)) {
            threadManager.executeThreadsByStrideGroups(targetIndices, processor, numberOfThreads);
        } else { // Default to sequential
            threadManager.executeThreadsBySliceGroups(targetIndices, processor, numberOfThreads);
        }
        return txt;
    }

    public static int[] FindHubs(AGN agn) {
        int nrhubs = agn.getNrgenes() / 10; //assuming that 10% of genes are hubs.
        int[] ihubs = new int[nrhubs];

        int[] genesdegree = new int[agn.getNrgenes()];
        for (int i = 0; i < agn.getNrgenes(); i++) {
            genesdegree[i] = agn.getGenes()[i].getPredictors().size() + agn.getGenes()[i].getTargets().size();
        }
        int[] orderedindexes = Preprocessing.BubbleSortDEC(genesdegree);
        for (int i = 0; i < nrhubs; i++) {
            ihubs[i] = orderedindexes[i];
        }
        return (ihubs);
    }

    //a informacao do target pode ser o Locus ou o probsetname.
    public static void FindIndexes(AGN agn, String[] targetinfo, Vector targetindexes) {
        for (int ind = 0; ind < targetinfo.length; ind++) {
            boolean found = false;
            int g = 0;
            String tinf = targetinfo[ind];
            int classe = -1;
            while (!found && g < agn.getNrgenes()) {
                if (tinf.equalsIgnoreCase(agn.getGenes()[g].getProbsetname()) || tinf.equalsIgnoreCase(agn.getGenes()[g].getLocus())) {
                    found = true;
                    if (agn.getGenes()[g].getName().equalsIgnoreCase("THI1")) {
                        agn.getGenes()[g].setClasse(4);
                    }
                    if (agn.getGenes()[g].getName().equalsIgnoreCase("LOS2")) {
                        agn.getGenes()[g].setClasse(6);
                    }
                    classe = agn.getGenes()[g].getClasse();
                } else {
                    g++;
                }
            }
            int index = (Integer) ((Vector) targetindexes.get(ind)).get(0);
            if (found && index == -1) {
                //targetindexes.setElementAt(g, ind);
                ((Vector) targetindexes.get(ind)).setElementAt(g, 0);
                ((Vector) targetindexes.get(ind)).setElementAt(classe, 1);
            }
        }
    }

    public static void AddAffymetrixInformation(AGN network, String pathinputfile) throws IOException {
        Vector collumns = new Vector(9);
        for (int i = 0; i < 9; i++) {
            collumns.add(i);
        }
        Vector[] geneinformations = IOFile.ReadDataCollumns(pathinputfile, 1, collumns, "\t");

        //debug
        //for (int i = 0; i < geneinformations.length; i++) {
        //    if (geneinformations[i].size() > 0) {
        //        IOFile.PrintlnAndLog((String) geneinformations[i].get(0));
        //    }
        //}
        //end-debug

        for (int g = 0; g < network.getNrgenes(); g++) {
            Gene gene = network.getGenes()[g];
            gene.setIndex(g);
            String geneprobsetname = gene.getProbsetname();
            for (int i = 0; i < geneinformations[0].size(); i++) {
                if (((String) geneinformations[0].get(i)).equalsIgnoreCase(geneprobsetname)) {
                    //IOFile.PrintlnAndLog((String) geneinformations[0].get(i));
                    //achou a referencia
                    String probsetname = (String) geneinformations[0].get(i);
                    String arrayelementtype = (String) geneinformations[1].get(i);
                    String organism = (String) geneinformations[2].get(i);
                    String iscontrol = (String) geneinformations[3].get(i);
                    String locus = (String) geneinformations[4].get(i);
                    String description = (String) geneinformations[5].get(i);
                    String chromosometype = (String) geneinformations[6].get(i);
                    String start = (String) geneinformations[7].get(i);
                    String stop = (String) geneinformations[8].get(i);

                    gene.setProbsetname(probsetname);
                    gene.setType(arrayelementtype);
                    gene.setLocus(locus);
                    gene.setOrganism(organism);
                    gene.setDescription(description);
                    if (iscontrol.equalsIgnoreCase("no")) {
                        gene.setControl(false);
                    } else if (iscontrol.equalsIgnoreCase("yes")) {
                        gene.setControl(true);
                    }
                    if (isNumber(start)) {
                        gene.setStart(Integer.valueOf(start));
                    }
                    if (isNumber(stop)) {
                        gene.setStop(Integer.valueOf(stop));
                    }
                    gene.setChromosometype(chromosometype);

                    /*
                    if (!description.equalsIgnoreCase("no_match")) {
                    //quebrar a descricao e atribuir o inicio como nome do gene.
                    String breakdescription = String.valueOf(';') + String.valueOf(')');
                    StringTokenizer s = new StringTokenizer(description, breakdescription, true);
                    String name = s.nextToken();
                    boolean par = false;
                    for (int c = 0; c < name.length(); c++) {
                    if (name.charAt(c) == '(') {
                    par = true;
                    c = name.length();
                    }
                    }
                    if (par) {
                    name = name + ")";
                    }
                    //IOFile.PrintlnAndLog(name);
                    gene.setName(name);
                    } else {
                    gene.setName(probsetname);
                    }*/
                    i = geneinformations[0].size();
                }
            }
        }
        //AGNRoutines.ViewAGNOLD(network);
        //IOFile.WriteAGNnewtoFile(network, pathnetwork + "new");
    }

    public static boolean isNumber(String str) {
        boolean bool = false;
        try {
            int num = Integer.parseInt(str);
            bool = true;
        } catch (NumberFormatException exception) {
            bool = false;
        }
        return bool;
    }

    public static void AddNCBIInformation(AGN network, String pathinputfile) throws IOException {
        String delimiter = String.valueOf(' ') + String.valueOf(',') + String.valueOf('=') + String.valueOf('.') + String.valueOf(')') + String.valueOf('(') + String.valueOf('\t') + String.valueOf('\n') + String.valueOf('\r') + String.valueOf('\f') + String.valueOf(';');
        String delimitersp = String.valueOf(' ') + String.valueOf('=') + String.valueOf('\t') + String.valueOf('\n') + String.valueOf('\r') + String.valueOf('\f') + String.valueOf(';');
        int start = -1;
        int stop = -1;
        String genename = null;
        String synonyms = "";
        String function = "";
        String locus = "";
        int geneid = -1;
        String notes = "";
        String product = "";
        String proteinid = "";
        int chromosome = -1;
        boolean found = false;
        int notetype = 0;
        int synonymscount = 0;
        int genecount = 0;
        Gene gene = null;
        BufferedReader br = IOFile.OpenBufferedReader(pathinputfile);
        while (br.ready()) {
            StringTokenizer s = new StringTokenizer(br.readLine(), delimiter);
            if (s.countTokens() > 0) {
                String token = s.nextToken();
                if (token.equalsIgnoreCase("gene")) {
                    if (found) {
                        //complete the informations
                        gene.setOrganism("Arabidopsis thaliana");
                        gene.setStart(start);
                        gene.setStop(stop);
                        gene.setName(genename);
                        gene.setSynonyms(synonyms);
                        gene.setFunction(function);
                        gene.setLocus(locus);
                        gene.setGeneid(geneid);
                        gene.setDescription(notes);
                        gene.setProduct(product);
                        gene.setProteinid(proteinid);
                        gene.setChromosome(chromosome);
                        IOFile.PrintlnAndLog("Found Locus = " + gene.getLocus() + " | ProbSetName = " + gene.getProbsetname());
                        found = false;
                        start = -1;
                        stop = -1;
                        genename = null;
                        synonyms = "";
                        function = "";
                        locus = "";
                        geneid = -1;
                        notes = "";
                        product = "";
                        proteinid = "";
                    }
                    genecount++;
                    while (s.hasMoreTokens()) {
                        String subtoken = s.nextToken();
                        if (subtoken.equalsIgnoreCase("complement")) {
                            start = Integer.valueOf(s.nextToken());
                            stop = Integer.valueOf(s.nextToken());
                        } else if (isNumber(subtoken)) {
                            start = Integer.valueOf(subtoken);
                            if (s.hasMoreTokens()) {
                                subtoken = s.nextToken();
                                if (isNumber(subtoken)) {
                                    stop = Integer.valueOf(subtoken);
                                }
                            }
                        }
                        //if (posini == 22246592) {
                        //IOFile.PrintlnAndLog("achou o THI1");
                        //}
                    }
                }
                if (token.equalsIgnoreCase("mRNA")) {
                    notetype = 1;
                }
                if (token.equalsIgnoreCase("CDS")) {
                    notetype = 2;
                }
                if (token.equalsIgnoreCase("/function")) {
                    function = s.nextToken(String.valueOf('='));
                    token = "";//inicializazao para entrar no while.
                    do {
                        s = new StringTokenizer(br.readLine(), delimitersp);
                        while (s.hasMoreTokens() && !token.startsWith("/")) {
                            token = s.nextToken();
                            if (!token.startsWith("/")) {
                                function += " " + token;
                            }
                        }
                    } while (!token.startsWith(String.valueOf("/")));
                    //IOFile.PrintlnAndLog("Function == " + function);
                }
                if (token.equalsIgnoreCase("/note") && notetype == 2) {
                    //considera apenas as anotacoes da CoDing Sequence (CDS).
                    notes = s.nextToken(String.valueOf('='));
                    token = "";//inicializazao para entrar no while.
                    do {
                        s = new StringTokenizer(br.readLine(), delimitersp);
                        while (s.hasMoreTokens() && !token.startsWith("/")) {
                            token = s.nextToken();
                            if (!token.startsWith("/")) {
                                notes += " " + token;
                            }
                        }
                    } while (!token.startsWith(String.valueOf("/")));
                    //IOFile.PrintlnAndLog("Notes == " + notes);
                }
                if (token.equalsIgnoreCase("/product")) {
                    product = s.nextToken(String.valueOf('='));
                    token = "";//inicializazao para entrar no while.
                    do {
                        s = new StringTokenizer(br.readLine(), delimitersp);
                        while (s.hasMoreTokens() && !token.startsWith("/")) {
                            token = s.nextToken();
                            if (!token.startsWith("/")) {
                                product += " " + token;
                            }
                        }
                    } while (!token.startsWith(String.valueOf("/")));
                    //IOFile.PrintlnAndLog("Product == " + product);
                }
                if (token.equalsIgnoreCase("/gene")) {
                    genename = s.nextToken(String.valueOf('"') + String.valueOf('"') + String.valueOf('='));
                    //IOFile.PrintlnAndLog(genename);
                }
                if (token.equalsIgnoreCase("/protein_id")) {
                    proteinid = s.nextToken(String.valueOf('"') + String.valueOf('"') + String.valueOf('='));
                    //IOFile.PrintlnAndLog(proteinid);
                }
                if (token.equalsIgnoreCase("/locus_tag")) {
                    locus = s.nextToken(String.valueOf('"') + String.valueOf('"') + String.valueOf('='));
                    if (!found) {
                        int g = 0;
                        while (g < network.getNrgenes() && !found) {
                            gene = network.getGenes()[g];
                            //correction of locus information
                            //in some cases, locus has two stick entries, separated by semicolon
                            String genelocus = gene.getLocus();
                            StringTokenizer st = new StringTokenizer(genelocus, String.valueOf(' ') + String.valueOf(';'));
                            genelocus = st.nextToken();
                            gene.setLocus(genelocus);
                            if (locus.equalsIgnoreCase(gene.getLocus())) {
                                found = true;
                            } else {
                                g++;
                            }
                        }
                    }
                    //IOFile.PrintlnAndLog(locus);
                    synonymscount = 0;
                }
                if (token.equalsIgnoreCase("/gene_synonym")) {
                    if (synonymscount == 0) {
                        synonyms = s.nextToken(String.valueOf('"') + String.valueOf('"') + String.valueOf('='));
                    } else {
                        synonyms += ("; " + s.nextToken(String.valueOf('"') + String.valueOf('"') + String.valueOf('=')));
                    }
                    //IOFile.PrintlnAndLog(synonyms);
                    synonymscount++;
                }
                if (token.equalsIgnoreCase("/db_xref")) {
                    String subtoken = s.nextToken(String.valueOf(' ') + String.valueOf('"') + String.valueOf(':') + String.valueOf('='));
                    if (subtoken.equalsIgnoreCase("GeneID")) {
                        geneid = Integer.valueOf(s.nextToken());
                        //IOFile.PrintlnAndLog(geneid);
                    }
                }
                if (token.equals("DEFINITION")) {
                    while (s.hasMoreTokens()) {
                        token = s.nextToken();
                        if (token.equalsIgnoreCase("chromosome")) {
                            chromosome = Integer.valueOf(s.nextToken());
                        }
                    }
                }
                //IOFile.PrintlnAndLog(token);
            }
        }
        br.close();
    }

    public static String FindPathwayDescription(String pathwaydescription, String pathway) throws IOException {
        String delimiter = String.valueOf(' ') + String.valueOf(',') + String.valueOf('=') + String.valueOf('.') + String.valueOf(')') + String.valueOf('(') + String.valueOf('\t') + String.valueOf('\n') + String.valueOf('\r') + String.valueOf('\f') + String.valueOf(';');
        String res = null;
        BufferedReader br = IOFile.OpenBufferedReader(pathwaydescription);
        boolean found = false;
        while (br.ready() && !found) {
            StringTokenizer s = new StringTokenizer(br.readLine(), delimiter);
            String pathwayid = s.nextToken();
            if (pathwayid.equalsIgnoreCase(pathway)) {
                found = true;
                res = "";
                while (s.hasMoreTokens()) {
                    res += s.nextToken() + " ";
                }
                res = res.substring(0, res.length() - 1);//retirar o ultimo espaco em branco adicionado no laco acima.
            }
        }
        return (res);
    }

    public static void AddKEEGInformation(
            AGN network,
            String pathwaydata,
            String pathwaydescription) throws IOException {
        String delimiter = String.valueOf(' ') + String.valueOf(',') + String.valueOf('=') + String.valueOf('.') + String.valueOf(')') + String.valueOf('(') + String.valueOf('\t') + String.valueOf('\n') + String.valueOf('\r') + String.valueOf('\f') + String.valueOf(';');
        Gene gene = null;
        BufferedReader br = IOFile.OpenBufferedReader(pathwaydata);
        while (br.ready()) {
            StringTokenizer s = new StringTokenizer(br.readLine(), delimiter);
            if (s.countTokens() > 0) {
                String locus = s.nextToken();
                int g = 0;
                boolean found = false;
                while (g < network.getNrgenes() && !found) {
                    gene = network.getGenes()[g];
                    //correction of locus information
                    //in some cases, locus has two stick entries, separated by semicolon
                    String genelocus = gene.getLocus();
                    StringTokenizer st = new StringTokenizer(genelocus, String.valueOf(' ') + String.valueOf(';'));
                    genelocus = st.nextToken();
                    gene.setLocus(genelocus);
                    if (locus.equalsIgnoreCase(gene.getLocus())) {
                        found = true;
                    } else {
                        g++;
                    }
                }
                if (found) {
                    Vector pathway = new Vector();
                    Vector description = new Vector();
                    while (s.hasMoreTokens()) {
                        String pw = s.nextToken();
                        String desc = FindPathwayDescription(pathwaydescription, pw);
                        pathway.add(pw);
                        description.add(desc);
                        IOFile.PrintlnAndLog(pw + " == " + desc);
                    }
                    //complete the pathway information
                    gene.setPathway(pathway);
                    gene.setPathwaydescription(description);
                }
            }
        }
        br.close();
    }

    public static Vector SinalPlotMA(int row, boolean showlegend,
            float maxvalue, int startcol, AGN agn, boolean showchart) {
        //codigo para gerar os dados para o grafico MultipleStepChart - usando CategoryDataset
        DefaultCategoryDataset[] datasets = new DefaultCategoryDataset[5];
        float[][] Mo = agn.getTemporalsignal();
        float[][] Mn = agn.getTemporalsignalnormalized();
        int[][] Mq = agn.getTemporalsignalquantized();
        for (int i = 0; i < datasets.length; i++) {
            datasets[i] = new DefaultCategoryDataset();
        }
        Gene gene = agn.getGenes()[row];
        String genename = gene.getName();
        if (genename == null) {
            genename = gene.getLocus();
            if (genename == null || genename.equalsIgnoreCase("no_match")) {
                genename = gene.getProbsetname();
            }
        }
        for (int col = 0; col < Mo[0].length; col++) {
            if ((int) Mo[row][col] != Preprocessing.skipvalue) {
                String featuretitle = "x axis";
                if (agn.getLabelstemporalsignal() != null) {
                    featuretitle = (String) agn.getLabelstemporalsignal().get(col + startcol);
                }
                datasets[0].addValue(Mo[row][col], genename, featuretitle);
                datasets[1].addValue(Mq[row][col], genename, featuretitle);
                datasets[2].addValue(Mn[row][col], genename, featuretitle);
                datasets[3].addValue(agn.getLowthreshold()[col], genename, featuretitle);
                datasets[4].addValue(agn.getHithreshold()[col], genename, featuretitle);
            }
        }
        return (Chart.MultipleStepChartOverlayedMA(datasets,
                "Time-Series Data", "Time", "Value",
                showlegend, maxvalue + 0.03f, -0.03f, showchart));
    }

    public static void AdjustGeneNames(AGN agn) {
        for (int i = 0; i < agn.getNrgenes(); i++) {
            Gene gene = agn.getGenes()[i];
            if (gene.getName() == null) {
                if (gene.getLocus() != null && !gene.getLocus().equalsIgnoreCase("no_match")) {
                    gene.setName(gene.getLocus());
                } else {
                    gene.setName(gene.getProbsetname());
                }
            }
        }
    }

    public static int MaxGeneFrequency(AGN network, Gene gene) {
        int freq = 1;
        //verifica a frequencia desse gene nos seus targets
        for (int t = 0; t < gene.getTargets().size(); t++) {
            int indextarget = (Integer) gene.getTargets().get(t);
            Gene target = network.getGenes()[indextarget];
            int ftemp = EdgeFrequency(target, gene);
            if (ftemp > freq) {
                freq = ftemp;
            }
        }
        //verifica a frequencia dos preditores desse gene
        if (gene.getPredictorsties() != null && gene.getPredictorsties().length > 0) {
            for (int tie = 0; tie < gene.getPredictorsties().length; tie++) {
                Vector predictors = gene.getPredictorsties()[tie];
                for (int p = 0; p < predictors.size(); p++) {
                    int indexpredictortied = (Integer) predictors.get(p);
                    Gene predictor = network.getGenes()[indexpredictortied];
                    int ftemp = EdgeFrequency(gene, predictor);
                    if (ftemp > freq) {
                        freq = ftemp;
                    }
                }
            }
        } else if (gene.getPredictors() != null && gene.getPredictors().size() > 0) {
            for (int p = 0; p < gene.getPredictors().size(); p++) {
                int indexpredictortied = (Integer) gene.getPredictors().get(p);
                Gene predictor = network.getGenes()[indexpredictortied];
                int ftemp = EdgeFrequency(gene, predictor);
                if (ftemp > freq) {
                    freq = ftemp;
                }
            }
        }
        return (freq);
    }

    public static int EdgeFrequency(Gene target, Gene predictor) {
        int freq = 1;
        if (target.getPredictorsties() != null && target.getPredictorsties().length > 0) {
            for (int tie = 0; tie < target.getPredictorsties().length; tie++) {
                Vector predictors = target.getPredictorsties()[tie];
                for (int p = 0; p < predictors.size(); p++) {
                    int predictortied = (Integer) predictors.get(p);
                    if (predictortied == predictor.getIndex()) {
                        freq++;
                    }
                }
            }
        } else {
            freq = 25;
        }
        //se nao houve nenhum empate, preditor = 25, se houve empates,
        //se houve empates, preditor = sua frequencia como preditor do target selcionado.
        return (freq);
    }

    //conta as classes armazenadas num objeto vetor (matriz 1x2) ordenado pela coluna 2, as quais sao armazenadas na posicao 1 de cada linha da matriz
    public static int CountClasses(Vector v) {
        int classes = 0;
        if (v != null) {
            classes++;
        }
        for (int i = 0; i < v.size() - 1; i++) {
            if ((Integer) ((Vector) v.get(i)).get(1) != (Integer) ((Vector) v.get(i + 1)).get(1) && (Integer) ((Vector) v.get(i)).get(1) != 0) {
                classes++;
            }
        }
        return (classes);
    }

    //conta as classes armazenadas num objeto vetor (matriz 1x2) ordenado pela coluna 2, as quais sao armazenadas na posicao 1 de cada linha da matriz
    public static float CountGenesperClasses(Vector v, int classe) {
        float count = 0;
        for (int i = 0; i < v.size(); i++) {
            if ((Integer) ((Vector) v.get(i)).get(1) == classe) {
                count++;
            }
        }
        return (count);
    }

    //return distinct patways that are inside network
    public static Vector Pathways(AGN agn) {
        Vector pathways = new Vector();
        for (int i = 0; i < agn.getGenes().length; i++) {
            Gene g = agn.getGenes()[i];
            if (g.getPathway() != null) {
                for (int p = 0; p < g.getPathway().size(); p++) {
                    String pathway = (String) g.getPathway().get(p);
                    if (!pathways.contains(pathway)) {
                        pathways.add(pathway);
                    }
                }
            }
        }
        return (pathways);
    }
}
