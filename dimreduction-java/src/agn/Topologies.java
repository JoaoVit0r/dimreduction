/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package agn;

import java.util.Random;
import java.util.Vector;

/**
 *
 * @author Fabricio
 */
public class Topologies {
    //usando espaco usado para geracao do modelo geografico.

    public static final int graphheight = 500;
    public static final int graphwidth = 500;

    public static void RemoveEdges(AGN agn) {
        Random rn = new Random(System.nanoTime());
        for (int it = 0; it < agn.getNrgenes(); it++) {
            Gene target = agn.getGenes()[it];
            for (int ip = 0; ip < target.getPredictors().size();) {
                int indexpredictor = (Integer) target.getPredictors().get(ip);
                Gene predictor = agn.getGenes()[indexpredictor];
                if (rn.nextDouble() > 0.5d) {
                    //remove aresta entre target e predictor.
                    target.removePredictor(indexpredictor);
                    predictor.removeTarget(it);
                } else {
                    ip++;
                }
            }
        }
    }

    public static void CreateLogicalCircuit(AGN agn) {
        //inicializa a geracao de pseudo-aleatorios utilizando a hora do sistema como semente.
        Random rnbf = new Random(System.nanoTime());//funcao booleana
        //Random rnnot = new Random(System.nanoTime());//funcao not
        for (int target = 0; target < agn.getNrgenes(); target++) {
            if (agn.getGenes()[target].getPredictors().size() > 0) {
                //funcoes Booleanas que serao definidas entre cada dois preditores.
                Vector booleanfunctions = new Vector(agn.getGenes()[target].getPredictors().size() - 1);
                //atribui uma funcao booleana ao gene alvo, se ele tive apenas um preditor.
                if (agn.getGenes()[target].getPredictors().size() == 1) {
                    booleanfunctions = new Vector(1);
                    int ch;
                    if (agn.isAllbooleanfunctions()) {
                        //escolhe uma entre 4 possiveis...caso especial.
                        ch = rnbf.nextInt(4);
                    } else {
                        //escolhe uma entre 2 possiveis...caso especial.
                        ch = rnbf.nextInt(2);
                    }
                    //escolha 0 == A, 1 == NOT A, 2 == CONTRADICTION, 3 == TAUTOLOGY
                    if (ch == 0) {
                        booleanfunctions.add(0, 11);
                    } else if (ch == 1) {
                        booleanfunctions.add(0, 14);
                    } else if (ch == 2) {
                        booleanfunctions.add(0, 10);
                    } else {
                        booleanfunctions.add(0, 15);
                    }
                } else {
                    for (int i = 0; i < agn.getGenes()[target].getPredictors().size() - 1; i++) {
                        //definicao da funcao Booleana para cada par de preditores de forma aleatoria.
                        if (agn.isAllbooleanfunctions()) {
                            booleanfunctions.add(i, rnbf.nextInt(16));
                        } else {
                            booleanfunctions.add(i, rnbf.nextInt(10));
                        }
                        //0  == A AND B
                        //1  == A AND NOT B
                        //2  == NOT A AND B
                        //3  == A XOR B
                        //4  == A OR B
                        //5  == A NOR B
                        //6  == A XNOR B
                        //7  == A OR NOT B
                        //8  == NOT A OR B
                        //9  == A NAND B
                        //10 == CONTRADICTION (always false)
                        //11 == A
                        //12 == B
                        //13 == NOT B
                        //14 == NOT A
                        //15 == TAUTOLOGY (always true)
                    }
                }
                agn.getGenes()[target].setBooleanfunctions(booleanfunctions);
                //A execucao das funcoes booleanas aos preditores e definicao
                //dos valores aos targets eh realizada na classe Simulation,
                //metodo ApplyLogicalCircuit() e ApplyBooleanFunction().
                //Estes metodos sao chamados a partir da classe AGN, metodo CreateTemporalSignal().
            }
        }
    }

    /*Metodo para definicao da arquitetura da rede, baseada no modelo de
    Erdos-Renyi (paper Luciano) */
    public static void MakeErdosRenyiTopology(AGN agn) {
        //inicializa a geracao de pseudo-aleatorios utilizando a hora do sistema como semente.
        Random rn = new Random(System.nanoTime());
        double prob = (double) agn.getAvgedges() / (agn.getNrgenes() - 1);
        //continuacao da definicao das regras.
        //laco para sortear os preditores de cada um dos genes target 't'.
        for (int target = 0; target < agn.getNrgenes(); target++) {
            //int target = i;//rn.nextInt(nrnodes);
            for (int predictor = 0; predictor < agn.getNrgenes(); predictor++) {
                //assumido que o numero de preditores para o target 't' tem que ser pelo menos 1.
                if (predictor != target && !agn.getGenes()[target].getPredictors().contains(predictor)) {
                    if (rn.nextDouble() < prob) {
                        //cria duas arestas target <--> predictor.
                        agn.getGenes()[target].addPredictor(predictor);
                        agn.getGenes()[predictor].addTarget(target);

                        agn.getGenes()[predictor].addPredictor(target);
                        agn.getGenes()[target].addTarget(predictor);
                    }
                }
            }
            //assumido que o target nao pode ser predito por ele mesmo.
            //nao permitida a auto-predicao e nem a duplicidade de predicao.
        }
    }

    /*Metodo para definicao da arquitetura da rede, baseada no modelo de
    Barabasi-Albert (paper Luciano) */
    public static void MakeBarabasiAlbertTopology(AGN agn) {
        int n0 = (int) (0.1 * agn.getNrgenes());//numero de nos para inicializacao da rede.
        //definicao do vetor para contar o nr de vezes que um gene eh selecionado como preditor de outro gene.
        Vector verticeslist = new Vector();
        //inicializa a geracao de pseudo-aleatorios utilizando a hora do sistema como semente.
        Random rn = new Random(System.nanoTime());
        if (agn.getNrgenes() < n0) {
            return;
            //vetor de preditores para os targets.
        }
        double prob = (double) agn.getAvgedges() / (n0 - 1);
        //inicializacao do grafo com m0 nos de forma aleatoria (Erdos-Renyi)
        for (int target = 0; target < n0; target++) {
            //int target = i;//rn.nextInt(nr_genes);
            for (int predictor = 0; predictor < n0; predictor++) {
                //assumido que o numero de preditores para o target 't' tem que ser pelo menos 1.
                //int predictor = j;//rn.nextInt(nr_genes);
                if (predictor != target && !agn.getGenes()[target].getPredictors().contains(predictor)) {
                    if (rn.nextDouble() < prob) {
                        //cria duas arestas target <--> predictor.
                        agn.getGenes()[target].addPredictor(predictor);
                        agn.getGenes()[predictor].addTarget(target);

                        agn.getGenes()[predictor].addPredictor(target);
                        agn.getGenes()[target].addTarget(predictor);

                        //adiciona predictor e target na lista de escolha
                        verticeslist.add(predictor);
                        verticeslist.add(target);
                    }
                }
            }
        }
        for (int target = n0; target < agn.getNrgenes(); target++) {
            while ((agn.getGenes()[target].getPredictors().size() +
                    agn.getGenes()[target].getTargets().size()) < 2 * agn.getAvgedges()) {
                int position = rn.nextInt(verticeslist.size());
                int predictor = (Integer) verticeslist.get(position);
                if (predictor != target && !agn.getGenes()[target].getPredictors().contains(predictor)) {
                    //cria duas arestas target <--> predictor.
                    agn.getGenes()[target].addPredictor(predictor);
                    agn.getGenes()[predictor].addTarget(target);

                    agn.getGenes()[predictor].addPredictor(target);
                    agn.getGenes()[target].addTarget(predictor);

                    //adiciona predictor e target na lista de escolha
                    verticeslist.add(predictor);
                    verticeslist.add(target);
                    //assumido que o target nao pode ser predito por ele mesmo.
                    //nao permitida a auto-predicao e nem a duplicidade de predicao.
                }
            }
        }
    }

    /* Metodo para definicao da arquitetura da rede, baseada no modelo de
    Geografico */
    public static void MakeGeographicalTopology(AGN agn) {//, int maxdistance) {
        //densidade == quantidade de vertices em cada pto do espaco.
        float density = (float) agn.getNrgenes() / (graphwidth * graphheight);
        //area quadrada estimada que serah ocupada por cada vertice da rede.
        //float areaquadradaporvertice = 1.0f / density;
        //distancia maxima para ligacao de arestas entre os vertices.
        //estah multiplicando por 2 pq esta gerando arestas bidirecionais que depois serao removidas.
        float maxdistance = (float) Math.sqrt((agn.getAvgedges() * 2) / (Math.PI * density));

        //sorteio das posicoes que serao ocupadas pelos vertices da rede.
        Random rnx = new Random(System.currentTimeMillis());
        try{
            Thread.sleep(20);
        }catch(InterruptedException error){
            //
        }
        Random rny = new Random(System.currentTimeMillis());
        for (int g = 0; g < agn.getNrgenes(); g++) {
            agn.getGenes()[g].setY((rny.nextFloat() * graphheight));
            agn.getGenes()[g].setX((rnx.nextFloat() * graphwidth));
        }
        for (int target = 0; target < agn.getNrgenes(); target++) {
            for (int predictor = 0; predictor < agn.getNrgenes(); predictor++) {
                double distance = CNMeasurements.EuclideanDistance(
                        agn.getGenes()[target],
                        agn.getGenes()[predictor]);
                if (distance <= maxdistance &&
                        predictor != target &&
                        !agn.getGenes()[target].getPredictors().contains(predictor)) {

                    //cria duas arestas target <--> predictor.
                    agn.getGenes()[target].addPredictor(predictor);
                    agn.getGenes()[predictor].addTarget(target);

                    agn.getGenes()[predictor].addPredictor(target);
                    agn.getGenes()[target].addTarget(predictor);
                }
            }
        }
    }

    /* max_predictors == numero maximo de vezes que um determinado gene pode
    ser preditor de outros genes.
    quantization == quantizacao dos valores assumidos para os features.
    net_architecture == "BA" -> Barab�si-Albert (scale-free)
    net_architecture == "ER" -> Erd�s-R�nyi (random)
    net_architecture == "GG" -> Geographical */
    public static AGN CreateNetwork(int nrgenes, int signalsize,
            int nrinitializations, float avgedges, int quantization,
            String topology, boolean allbooleanfunctions) {
        AGN agn = null;
        //double soma = 0;
        //int i = 0;
        //for (; i < 100; i++) {
        agn = new AGN(topology, nrgenes, signalsize, nrinitializations,
                quantization, avgedges, allbooleanfunctions);

        //define a topologia usando arestas bi-direcionais.
        if (topology.equalsIgnoreCase("BA")) {
            MakeBarabasiAlbertTopology(agn);
        } else if (topology.equalsIgnoreCase("ER")) {
            MakeErdosRenyiTopology(agn);
        } else if (topology.equalsIgnoreCase("GG")) {
            MakeGeographicalTopology(agn);
        } else {
            //modelo de rede nao especificado.
            return (null);
        }

        //float[] iodegree = CNMeasurements.AverageDegrees(agn.getGenes());
        //System.out.println("Grau medio = " + iodegree[0]);
        //agn.ViewAGN();

        //remove as arestas com probabilidade de 50%
        RemoveEdges(agn);

        //iodegree = CNMeasurements.AverageDegrees(agn.getGenes());
        //System.out.println("Novo grau medio = " + iodegree[0]);
        //agn.ViewAGN();
        //soma += iodegree[0];

        //define as funcoes de transicao para cada gene
        CreateLogicalCircuit(agn);
        //}
        //System.out.println("Execucoes = " + i);
        //System.out.println("Media = " + soma / i);
        return (agn);
    }
}
