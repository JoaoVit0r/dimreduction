/***************************************************************************/
/*** Interactive Graphic Environment for Dimensionality Reduction        ***/
/***                                                                     ***/
/*** Copyright (C) 2006  David Corrêa Martins Junior                     ***/
/***                     Fabrício Martins Lopes                          ***/
/***                     Roberto Marcondes Cesar Junior                  ***/
/***                                                                     ***/
/*** This library is free software; you can redistribute it and/or       ***/
/*** modify it under the terms of the GNU Lesser General Public          ***/
/*** License as published by the Free Software Foundation; either        ***/
/*** version 2.1 of the License, or (at your option) any later version.  ***/
/***                                                                     ***/
/*** This library is distributed in the hope that it will be useful,     ***/
/*** but WITHOUT ANY WARRANTY; without even the implied warranty of      ***/
/*** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   ***/
/*** Lesser General Public License for more details.                     ***/
/***                                                                     ***/
/*** You should have received a copy of the GNU Lesser General Public    ***/
/*** License along with this library; if not, write to the Free Software ***/
/*** Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA       ***/
/*** 02110-1301  USA                                                     ***/
/***                                                                     ***/
/*** Contact: David Corr�a Martins Junior - davidjr@vision.ime.usp.br    ***/
/***          Fabr�cio Martins Lopes - fabriciolopes@vision.ime.usp.br   ***/
/***          Roberto Marcondes Cesar Junior - cesar@vision.ime.usp.br   ***/
/***************************************************************************/
/***************************************************************************/
/*** This class implements graph generation and a method for save it as  ***/
/*** image.                                                              ***/
/***************************************************************************/
package charts;

import agn.AGN;
import agn.AGNRoutines;
import com.jgraph.layout.JGraphFacade;
import com.jgraph.layout.JGraphLayout;
import com.jgraph.layout.graph.JGraphSimpleLayout;
import com.jgraph.layout.organic.JGraphFastOrganicLayout;
import com.jgraph.layout.organic.JGraphOrganicLayout;
import com.jgraph.layout.organic.JGraphSelfOrganizingOrganicLayout;
import com.jgraph.layout.tree.JGraphRadialTreeLayout;
import com.jgraph.layout.tree.JGraphTreeLayout;
import java.awt.Color;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import org.jgraph.JGraph;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.DefaultPort;
import org.jgraph.graph.GraphConstants;
import utilities.IOFile;

public class GraphDR extends javax.swing.JFrame {

    public static final int w = 800;
    public static final int h = 600;
    public AGN agn;
    public JGraph graph;
    public static final Rectangle2D area = new Rectangle2D.Float(15, 15, GraphDR.w - 15, GraphDR.h - 100);
    private JScrollPane scrollpane;

    public GraphDR(String name, JGraph graph, AGN agn) {
        this.setSize(w, h);
        this.setTitle(name);
        this.graph = graph;
        this.agn = agn;
        //initComponents();
        scrollpane = new JScrollPane(this.graph);
        this.getContentPane().add(scrollpane);
        java.awt.Dimension screenSize = java.awt.Toolkit.getDefaultToolkit().getScreenSize();
        this.setBounds((screenSize.width - w) / 2, (screenSize.height - h) / 2, w, h);
        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);

        GroupLayout = new javax.swing.ButtonGroup();
        jMenuBar1 = new javax.swing.JMenuBar();
        jMenu1 = new javax.swing.JMenu();
        jMenuItem1 = new javax.swing.JMenuItem();
        jMenuItem2 = new javax.swing.JMenuItem();
        jMenu2 = new javax.swing.JMenu();
        jRadioButtonMenuItem2 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem4 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem5 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem3 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem1 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem7 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem6 = new javax.swing.JRadioButtonMenuItem();

        jMenu1.setText("Network");

        jMenuItem1.setText("Save Network as Image");
        jMenuItem1.addActionListener(new java.awt.event.ActionListener() {

            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jMenuItem1ActionPerformed(evt);
            }
        });
        jMenu1.add(jMenuItem1);
        jMenuItem1.getAccessibleContext().setAccessibleName("Salvar Imagem");

        jMenuItem2.setText("Save Network");
        jMenuItem2.addActionListener(new java.awt.event.ActionListener() {

            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jMenuItem2ActionPerformed(evt);
            }
        });
        jMenu1.add(jMenuItem2);
        jMenuBar1.add(jMenu1);
        jMenu2.setText("Layout");

        GroupLayout.add(jRadioButtonMenuItem2);
        jRadioButtonMenuItem2.setSelected(true);
        jRadioButtonMenuItem2.setText("Circle");
        jRadioButtonMenuItem2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem2ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem2);

        GroupLayout.add(jRadioButtonMenuItem4);
        jRadioButtonMenuItem4.setText("Fast Organic");
        jRadioButtonMenuItem4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem4ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem4);

        GroupLayout.add(jRadioButtonMenuItem5);
        jRadioButtonMenuItem5.setText("Organic");
        jRadioButtonMenuItem5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem5ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem5);

        GroupLayout.add(jRadioButtonMenuItem3);
        jRadioButtonMenuItem3.setText("Radial Tree");
        jRadioButtonMenuItem3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem3ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem3);

        GroupLayout.add(jRadioButtonMenuItem1);
        jRadioButtonMenuItem1.setText("Random");
        jRadioButtonMenuItem1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem1ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem1);

        GroupLayout.add(jRadioButtonMenuItem7);
        jRadioButtonMenuItem7.setText("Self Organizing Organic");
        jRadioButtonMenuItem7.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem7ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem7);

        GroupLayout.add(jRadioButtonMenuItem6);
        jRadioButtonMenuItem6.setText("Tree");
        jRadioButtonMenuItem6.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem6ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem6);

        jMenuBar1.add(jMenu2);

        setJMenuBar(jMenuBar1);
    }

    public boolean SaveGraphPositions() {
        if (this.agn != null) {
            boolean refresh = AGNRoutines.RefreshGraphPositionsonAGN(this.agn, this.graph);
            if (refresh) {
                String path = IOFile.SaveAGNFile();
                if (path != null) {
                    IOFile.WriteAGNtoFile(this.agn, path);
                }
                JOptionPane.showMessageDialog(this, "Network has been saved successfully!", "Information", JOptionPane.INFORMATION_MESSAGE);
                return (true);
            }
        }
        JOptionPane.showMessageDialog(this, "Error on writing the network. The network was not saved.", "Error", JOptionPane.ERROR_MESSAGE);
        return (false);
    }

    public boolean SaveGraphasImage() {
        if (this.agn != null) {
            String path = IOFile.SaveIMGFile();
            if (path != null) {
                try {
                    OutputStream out = new BufferedOutputStream(new FileOutputStream(path));
                    String extension = IOFile.getExtension(path);
                    JGraph graph1 = graph;
                    Color bg = graph1.getBackground();
                    BufferedImage img = graph1.getImage(bg, 0);
                    ImageIO.write(img, extension, out);
                    out.flush();
                    out.close();
                    JOptionPane.showMessageDialog(this, "Image has been saved successfully!", "Information", JOptionPane.INFORMATION_MESSAGE);
                    return (true);
                } catch (IOException erro) {
                    JOptionPane.showMessageDialog(this, "Error on writing the image. The image was not saved.", "Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        }
        return (false);
    }

    public static DefaultGraphCell createVertex(String name, double x,
            double y, double w, double h, Color bg, boolean raised,
            boolean autosize) {
        // Create vertex with the given name
        DefaultGraphCell cell = new DefaultGraphCell(name);
        // Set bounds
        GraphConstants.setBounds(cell.getAttributes(),
                new Rectangle2D.Double(x, y, w, h));
        // Set fill color
        if (bg != null) {
            GraphConstants.setGradientColor(
                    cell.getAttributes(), Color.orange);
            GraphConstants.setOpaque(
                    cell.getAttributes(), true);
        }
        // Set raised border
        if (raised) {
            GraphConstants.setBorder(
                    cell.getAttributes(),
                    BorderFactory.createRaisedBevelBorder());
        } else // Set black border
        {
            GraphConstants.setBorderColor(
                    cell.getAttributes(), Color.black);
        }

        GraphConstants.setAutoSize(cell.getAttributes(), autosize);

        // Add a Port
        DefaultPort port = new DefaultPort();
        cell.add(port);
        port.setParent(cell);

        return cell;
    }

    public void RefreshLayout(JGraphLayout layout){
        JGraphFacade facade = new JGraphFacade(graph); // Pass the facade the JGraph instance
        facade.setIgnoresUnconnectedCells(true);
        facade.setDirected(true);
        //JGraphLayout layout = new JGraphSelfOrganizingOrganicLayout();
        //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_TILT,GraphDR.w-15,GraphDR.h-100);
        //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_RANDOM,GraphDR.w-15,GraphDR.h-100);
        //JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_CIRCLE,GraphDR.w-15,GraphDR.h-150);
        //JGraphLayout layout = new JGraphRadialTreeLayout(); // Create an instance of the appropriate layout
        //JGraphLayout layout = new JGraphFastOrganicLayout(); // Create an instance of the appropriate layout
        //JGraphLayout layout = new JGraphOrganicLayout(area); // Create an instance of the appropriate layout
        //JGraphLayout layout = new JGraphTreeLayout(); // Create an instance of the appropriate layout
        layout.run(facade); // Run the layout on the facade.
        facade.scale(area);
        Map nested = facade.createNestedMap(true, true); // Obtain a map of the resulting attribute changes from the facade
        graph.getGraphLayoutCache().edit(nested); // Apply the results to the actual graph
        //this.getContentPane().add(new JScrollPane(this.graph));
        //this.getContentPane().remove(scrollpane);
        //scrollpane = new JScrollPane(this.graph);
        //this.getContentPane().add(scrollpane);
        //this.repaint();
        scrollpane.setViewportView(graph);
    }


    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        GroupLayout = new javax.swing.ButtonGroup();
        jMenuBar1 = new javax.swing.JMenuBar();
        jMenu1 = new javax.swing.JMenu();
        jMenuItem1 = new javax.swing.JMenuItem();
        jMenuItem2 = new javax.swing.JMenuItem();
        jMenu2 = new javax.swing.JMenu();
        jRadioButtonMenuItem2 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem4 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem5 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem3 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem1 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem7 = new javax.swing.JRadioButtonMenuItem();
        jRadioButtonMenuItem6 = new javax.swing.JRadioButtonMenuItem();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                formWindowClosing(evt);
            }
        });
        getContentPane().setLayout(null);

        jMenu1.setText("Network");

        jMenuItem1.setText("Save Network as Image");
        jMenuItem1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jMenuItem1ActionPerformed(evt);
            }
        });
        jMenu1.add(jMenuItem1);
        jMenuItem1.getAccessibleContext().setAccessibleName("Salvar Imagem");

        jMenuItem2.setText("Save Network");
        jMenuItem2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jMenuItem2ActionPerformed(evt);
            }
        });
        jMenu1.add(jMenuItem2);

        jMenuBar1.add(jMenu1);

        jMenu2.setText("Layout");

        GroupLayout.add(jRadioButtonMenuItem2);
        jRadioButtonMenuItem2.setSelected(true);
        jRadioButtonMenuItem2.setText("Circle");
        jRadioButtonMenuItem2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem2ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem2);

        GroupLayout.add(jRadioButtonMenuItem4);
        jRadioButtonMenuItem4.setText("Fast Organic");
        jRadioButtonMenuItem4.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem4ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem4);

        GroupLayout.add(jRadioButtonMenuItem5);
        jRadioButtonMenuItem5.setText("Organic");
        jRadioButtonMenuItem5.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem5ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem5);

        GroupLayout.add(jRadioButtonMenuItem3);
        jRadioButtonMenuItem3.setText("Radial Tree");
        jRadioButtonMenuItem3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem3ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem3);

        GroupLayout.add(jRadioButtonMenuItem1);
        jRadioButtonMenuItem1.setText("Random");
        jRadioButtonMenuItem1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem1ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem1);

        GroupLayout.add(jRadioButtonMenuItem7);
        jRadioButtonMenuItem7.setText("Self Organizing Organic");
        jRadioButtonMenuItem7.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem7ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem7);

        GroupLayout.add(jRadioButtonMenuItem6);
        jRadioButtonMenuItem6.setText("Tree");
        jRadioButtonMenuItem6.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRadioButtonMenuItem6ActionPerformed(evt);
            }
        });
        jMenu2.add(jRadioButtonMenuItem6);

        jMenuBar1.add(jMenu2);

        setJMenuBar(jMenuBar1);

        java.awt.Dimension screenSize = java.awt.Toolkit.getDefaultToolkit().getScreenSize();
        setBounds((screenSize.width-800)/2, (screenSize.height-600)/2, 800, 600);
    }// </editor-fold>//GEN-END:initComponents

    private void jMenuItem1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jMenuItem1ActionPerformed
        SaveGraphasImage();
    }//GEN-LAST:event_jMenuItem1ActionPerformed

    private void jMenuItem2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jMenuItem2ActionPerformed
        SaveGraphPositions();
    }//GEN-LAST:event_jMenuItem2ActionPerformed

    private void formWindowClosing(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_formWindowClosing
    }//GEN-LAST:event_formWindowClosing

    private void jRadioButtonMenuItem2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRadioButtonMenuItem2ActionPerformed
        JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_CIRCLE,GraphDR.w-15,GraphDR.h-150);
        RefreshLayout(layout);
    }//GEN-LAST:event_jRadioButtonMenuItem2ActionPerformed

    private void jRadioButtonMenuItem4ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRadioButtonMenuItem4ActionPerformed
        JGraphLayout layout = new JGraphFastOrganicLayout(); // Create an instance of the appropriate layout
        RefreshLayout(layout);
    }//GEN-LAST:event_jRadioButtonMenuItem4ActionPerformed

    private void jRadioButtonMenuItem5ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRadioButtonMenuItem5ActionPerformed

        JGraphLayout layout = new JGraphOrganicLayout(area); // Create an instance of the appropriate layout
        RefreshLayout(layout);
    }//GEN-LAST:event_jRadioButtonMenuItem5ActionPerformed

    private void jRadioButtonMenuItem3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRadioButtonMenuItem3ActionPerformed
        JGraphLayout layout = new JGraphRadialTreeLayout(); // Create an instance of the appropriate layout
        RefreshLayout(layout);
    }//GEN-LAST:event_jRadioButtonMenuItem3ActionPerformed

    private void jRadioButtonMenuItem1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRadioButtonMenuItem1ActionPerformed
        JGraphLayout layout = new JGraphSimpleLayout(JGraphSimpleLayout.TYPE_RANDOM,GraphDR.w-15,GraphDR.h-100);
        RefreshLayout(layout);
    }//GEN-LAST:event_jRadioButtonMenuItem1ActionPerformed

    private void jRadioButtonMenuItem7ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRadioButtonMenuItem7ActionPerformed
        JGraphLayout layout = new JGraphSelfOrganizingOrganicLayout();
        RefreshLayout(layout);
    }//GEN-LAST:event_jRadioButtonMenuItem7ActionPerformed

    private void jRadioButtonMenuItem6ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRadioButtonMenuItem6ActionPerformed
        JGraphLayout layout = new JGraphTreeLayout(); // Create an instance of the appropriate layout
        RefreshLayout(layout);
    }//GEN-LAST:event_jRadioButtonMenuItem6ActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.ButtonGroup GroupLayout;
    private javax.swing.JMenu jMenu1;
    private javax.swing.JMenu jMenu2;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JMenuItem jMenuItem1;
    private javax.swing.JMenuItem jMenuItem2;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItem1;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItem2;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItem3;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItem4;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItem5;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItem6;
    private javax.swing.JRadioButtonMenuItem jRadioButtonMenuItem7;
    // End of variables declaration//GEN-END:variables
}
