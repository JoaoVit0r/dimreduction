/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package agn;

import fs.Preprocessing;
import java.awt.Color;
import java.io.IOException;
import java.util.Vector;
import utilities.IOFile;

/**
 * @author Fabricio
 */
public class MainMarieAnne {
    //usado para coloracao dos vertices do grafo

    public static final int[] RGBColors = new int[]{
        new Color(250, 250, 250).getRGB(),
        Color.RED.getRGB(),
        Color.PINK.getRGB(), 
        Color.GREEN.getRGB(),
        Color.BLUE.getRGB(),
        Color.RED.getRGB(), 
        Color.ORANGE.getRGB(),
        Color.LIGHT_GRAY.getRGB(),
        Color.MAGENTA.getRGB()};
    public static final Color[] colors = new Color[]{Color.RED,
        Color.PINK, Color.GREEN, Color.BLUE, Color.RED, Color.ORANGE,
        Color.LIGHT_GRAY, Color.LIGHT_GRAY};
    public static final String[] classenames = new String[]{"síntese de tiamina",
        "controle", "fotossíntese", "respiração", "síntese de tiamina",
        "glicólise", "sem classe", "sem classe"};
    public static final String delimiter = String.valueOf(' ') + String.valueOf('\t') + String.valueOf('\n') + String.valueOf('\r') + String.valueOf('\f') + String.valueOf(';');

    public static Color getBGColor(Gene g) {
        return getBGColor(g.getClasse());
    }

    public static Color getBGColor(int classe) {
        if (classe >= 0 && classe < colors.length) {
            return colors[classe];
        } else {
            return (new Color(250, 250, 250));
        }
    }

    public static Color getFGColor(int classe) {
        if (classe >= 0 && classe < colors.length) {
            return (colors[classe]);
        } else {
            return (Color.BLACK);
        }
    }

    public static String getClassName(int classe) {
        if (classe >= 0 && classe < classenames.length) {
            return (classenames[classe]);
        } else {
            return ("sem classe definida");
        }
    }

    public static int getRGB(Gene g) {
        Color cor = Color.YELLOW;
        if (g.getClasse() == 0) {
            cor = Color.RED;
        } else if (g.getClasse() == 1) {
            cor = Color.PINK;
        } else if (g.getClasse() == 2) {
            cor = Color.GREEN;
        } else if (g.getClasse() == 3) {
            cor = Color.BLUE;
        } else if (g.getClasse() == 4) {
            cor = Color.RED;
        } else if (g.getClasse() == 5) {
            cor = Color.ORANGE;
        } else if (g.getClasse() == 6) {
            cor = Color.LIGHT_GRAY;
        } else if (g.getClasse() == 7) {
            cor = Color.LIGHT_GRAY;
        }
        return cor.getRGB();
    }

    public static Vector[] RefreshGeneIDs(Vector remaingenes, Vector[] geneids) {
        Vector[] newgeneids = new Vector[geneids.length];
        newgeneids[0] = new Vector();
        newgeneids[1] = new Vector();
        for (int i = 0; i < remaingenes.size(); i++) {
            int lin = (Integer) remaingenes.get(i);
            newgeneids[0].add(i, geneids[0].get(lin));//atribui os novos ids que passaram pelo filtro.
            newgeneids[1].add(i, geneids[1].get(lin));//atribui os novos ids que passaram pelo filtro.
        }
        return newgeneids;
    }

    public static int getMetabolicPathway(Gene g){
        int classe = 8;
        for (int i = 0; i < g.getPathway().size(); i++){
            String pws = (String) g.getPathway().get(i);
            if (pws.equalsIgnoreCase("00730")){//CODIGO KEGG PARA TIAMINA
                classe = 5;
            }else if (pws.equalsIgnoreCase("00010")){//CODIGO KEGG PARA GLICOLISE
                classe = 6;
            }else if (pws.equalsIgnoreCase("03420")){//CODIGO KEGG PARA CONTROLE
                classe = 2;
            }
        }
        return(classe);
    }

    public static void main(String[] args) throws IOException {
        String[] entrada = {
            "dados-root-frio.csv",//0
            "dados-root-normal.csv",//1
            "dados-shoot-frio.csv",//2
            "dados-shoot-normal.csv",//3
            "dados-razoes-frio-normal-root.csv",//4
            "dados-razoes-frio-normal-shoot.csv"};//5
        String[] saida = {
            "agn-dados-root-frio.agn",
            "agn-dados-root-normal.agn",
            "agn-dados-shoot-frio.agn",
            "agn-dados-shoot-normal.agn",
            "agn-dados-razoes-frio-normal-root.agn",
            "agn-dados-razoes-frio-normal-shoot.agn"};
        //nohup ~/jdk1.6.0_16/bin/java -jar dimreductionMA.jar -v ../../Marie-Anne/dados-frio/ ../../Marie-Anne/html-files/ 1 > ../../Marie-Anne/html-files/saida-exe-dados-root-normal.txt &
        //nohup ~/jdk1.6.0_16/bin/java -jar dimreductionMA.jar -v ../../Marie-Anne/dados-frio/ resultadosMA-normalization/root-cold/ 0 > resultadosMA-normalization/root-cold/saida-exe-root-frio.txt &

        //parameters of the method.
        String type_entropy = "no_obs"; //tipo de penalizacao aplicada. no_obs ou poor_obs
        float q_entropy = 1;            //if selected criterion function is CoD, q_entropy = 0. Any else float value == Entropy.
        float alpha = 1;
        float beta = 0.8f;
        int sizeresultlist = 1;         //MAXIMO DE RESPOSTAS DA SELACAO DE CARACTERISTICAS.
        int quantization = 2;

        if (args.length > 0) {
            if (args[0].equalsIgnoreCase("-v")) {
                //-v D:\doutorado\Marie-Anne\dados-frio\ D:\doutorado\Marie-Anne\dados-frio\novos-resultados-redes-frio\dados-razoes-frio-normal-root\ 4
                String pathinput = args[1]; //arquivo contendo os dados de entrada.
                String pathoutput = args[2];//pasta para escrita dos resultados.
                int exe = Integer.valueOf(args[3]);//file index
                int maxfeatures =Integer.valueOf(args[4]);//max features to expand the search

                boolean log2 = false;//Boolean.parseBoolean(args[4]);//apply log2 on microarray values.
                String pathinputfile = pathinput + entrada[exe];
                //read data
                float[][] expressiondata = IOFile.ReadMatrix(pathinputfile, 1, 2, delimiter);
                Vector featurestitles = IOFile.ReadDataFirstRow(pathinputfile, 0, 0, delimiter);
                Vector collumns = new Vector(2);
                collumns.add(0);
                collumns.add(1);
                Vector[] geneids = IOFile.ReadDataCollumns(pathinputfile, 1, collumns, delimiter);
                Vector removedgenes = new Vector();
                Vector remaingenes = new Vector();

                //remove missing values on data.
                //retira os genes que apresentem algum spot com valor 0.
                float[][] filtereddata = Preprocessing.FilterMA(expressiondata, geneids, remaingenes, removedgenes);
                IOFile.WriteMatrix(pathoutput + "filtered-" + entrada[exe], filtereddata, ";");
                //refresh geneIDs information, removing filtered rows.
                geneids = RefreshGeneIDs(remaingenes, geneids);

                //apply log2 function on filtered data.
                if (log2) {
                    filtereddata = Preprocessing.ApplyLog2(filtereddata);
                }

                int nrgenes = filtereddata.length;
                int signalsize = filtereddata[0].length;
                float[] mean = new float[signalsize];
                float[] std = new float[signalsize];
                float[] lowthreshold = new float[signalsize];
                float[] hithreshold = new float[signalsize];
                int[][] quantizeddata = new int[nrgenes][signalsize];
                //data quantization
                float[][] normalizeddata = Preprocessing.quantizecolumnsMAnormal(
                        filtereddata,
                        quantizeddata,
                        3,
                        mean,
                        std,
                        lowthreshold,
                        hithreshold);
                IOFile.WriteMatrix(pathoutput + "normalized-" + entrada[exe], normalizeddata, ";");
                IOFile.WriteMatrix(pathoutput + "quantized-" + entrada[exe], quantizeddata, ";");

                //REMOVER DADOS QUE APRESENTAM PADRAO CONSTANTE NA MATRIZ...
                //nova filtragem apos a quantizacao...

                //create data structures and keep used informations about the data.
                AGN recoverednetwork = new AGN(nrgenes, signalsize, quantization);
                recoverednetwork.setMean(mean);
                recoverednetwork.setStd(std);
                recoverednetwork.setRemovedgenes(removedgenes);
                AGNRoutines.setPSNandClass(recoverednetwork, geneids);
                recoverednetwork.setTemporalSignal(filtereddata, featurestitles);
                recoverednetwork.setTemporalsignalquantized(quantizeddata);
                recoverednetwork.setTemporalsignalnormalized(normalizeddata);
                recoverednetwork.setLowthreshold(lowthreshold);
                recoverednetwork.setHithreshold(hithreshold);

                //add biological and structural informations about the genes from two sources.
                AGNRoutines.AddAffymetrixInformation(recoverednetwork,
                        pathinput + "affy_ATH1_array_elements-2009-7-29.txt");
                AGNRoutines.AddNCBIInformation(recoverednetwork,
                        pathinput + "NC_003070-003076.gbk");
                AGNRoutines.AddKEEGInformation(
                        recoverednetwork,
                        pathinput + "ath_gene_map.tab",
                        pathinput + "map_title.tab");

                //adjust the name of the genes
                AGNRoutines.AdjustGeneNames(recoverednetwork);
                IOFile.WriteAGNtoFile(recoverednetwork, pathoutput + "data-" + saida[exe]);

                //INICIO-DEBUG
                //int exe = 2;
                //String pathoutput = "/media/arquivos/doutorado/Marie-Anne/resultados-arabidopsis/dados-shoot-frio/";
                //recoverednetwork = IOFile.ReadAGNfromFile(pathoutput+"data-" + saida[exe]);
                //FIM-DEBUG

                String[] targetlocus = {
                    "AT5G54770",
                    "AT4G34200",
                    "AT2G36530",
                    "AT5G41370",
                    "AT1G05055",
                    "AT1G03190",
                    "AT3G05210",
                    "AT1G14030",
                    "AT1G67090",
                    "AT2G28000",
                    "AT2G34590",
                    "AT3G55410",
                    "AT4G24620",
                    "AT2G22480",
                    "AT2G01140",
                    "AT1G74030",
                    "AT4G37870",
                    "AT3G04080",
                    "AT1G22940",
                    "AT3G24030",
                    "AT5G65720",
                    "AT1G09430",
                    "AT2G47510",
                    "AT1G01090"};

                Vector targetindexes = new Vector();
                for (int tl = 0; tl < targetlocus.length; tl++) {
                    targetindexes.add(-1);
                }

                //recover index of the target genes taking into account the Locus.
                AGNRoutines.FindIndexes(recoverednetwork, targetlocus, targetindexes);

                //define the seed genes, which will be used to recover the network.
                String[] targetPSNs = {
                    "248128_AT",
                    "253274_AT",
                    "263924_AT",
                    "249307_S_AT",
                    "265218_AT",
                    "264356_AT",
                    "259304_AT",
                    "262648_AT",
                    "264474_S_AT",
                    "264069_AT",
                    "266904_AT",
                    "251787_AT",
                    "254141_AT",
                    "264044_AT",
                    "265735_AT",
                    "260392_AT",
                    "253041_AT",
                    "258567_AT",
                    "264771_AT",
                    "256907_AT",
                    "247164_AT",
                    "264504_AT",
                    "248461_S_AT",
                    "261583_AT"};

                //recover index of the target genes taking into account the ProbSetNames.
                AGNRoutines.FindIndexes(recoverednetwork, targetPSNs, targetindexes);
                System.out.println("Numero Total de targets = " + targetindexes.size());

                //transform target indexes in other data strucuture, wich will be submitted to network identification method.
                Vector vtargets = new Vector();

                //INICIO-DEBUG
                //vtargets.add(String.valueOf(18094));
                //FIM-DEBUG

                for (int gt = 0; gt < targetindexes.size(); gt++) {
                    int tindex = (Integer) targetindexes.get(gt);
                    //remove os targets nao encontrados.
                    if (tindex >= 0) {
                        vtargets.add(String.valueOf(tindex));
                    }
                    System.out.println(targetlocus[gt] + " == " +
                            targetPSNs[gt] + " == " +
                            recoverednetwork.getGenes()[tindex].getName() + " == " +
                            tindex);
                }
                System.out.println("Numero de targets encontrados = " + vtargets.size());

                //DEBUG
                //AGN recoverednetwork = IOFile.ReadAGNnewfromFile("D:/doutorado/dimreduction/dist/resultadosMA-normalization/razoes-shoot/data-agn-dados-razoes-frio-normal-shoot.agnnew");
                //FIM-DEBUG

                //identification of network from target genes.
                AGNRoutines.RecoverNetworkfromTemporalExpression(
                        recoverednetwork,
                        null,
                        1, //datatype: 1==temporal, 2==steady-state.
                        false,
                        1,//threshold_entropy
                        type_entropy,
                        alpha,
                        beta,
                        q_entropy,
                        vtargets, //targets
                        maxfeatures,
                        4,//SFFS_stack(pilha)//1==SFS, 2==Exhaustive, 3==SFFS, 4==SFFS_stack(expandindo todos os empates encontrados).
                        false,//jCB_TargetsAsPredictors.isSelected()
                        sizeresultlist,
                        null,
                        "sequential",
                        1);

                //armazenamento dos resultados
                IOFile.WriteAGNtoFile(recoverednetwork, pathoutput + "complete-" + saida[exe]);

                //INICIO-DEBUG
                //AGN recoverednetwork = IOFile.ReadAGNnewfromFile(pathoutput + "complete-" + saida[exe]);
                //FIM-DEBUG

                BuildHTML.BuildIndexPage(
                        recoverednetwork,
                        pathoutput,
                        entrada[exe].substring(0, entrada[exe].length() - 4),
                        vtargets);
                BuildHTML.BuildFiles(
                        recoverednetwork,
                        pathoutput,
                        vtargets);
            } else {
                //EXECUCAO COM INTERFACE GRAFICA.
                java.awt.EventQueue.invokeLater(new Runnable() {

                    @Override
                    public void run() {
                        new MainAGNWindow().setVisible(true);
                    }
                });
            }
        } else {
            //EXECUCAO COM INTERFACE GRAFICA.
            java.awt.EventQueue.invokeLater(new Runnable() {

                @Override
                public void run() {
                    new MainAGNWindow().setVisible(true);
                }
            });
        }
    }
}
