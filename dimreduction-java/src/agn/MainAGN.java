/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package agn; 

import java.io.File;
import java.io.IOException;
import utilities.IOFile; 

/**
 *
 * @author Fabricio
 */
public class MainAGN {

    public static void main(String[] args) throws IOException {
        if (args.length > 0) {
            if (args[0].equalsIgnoreCase("-v")) {
                java.util.Random rn = new java.util.Random();
                //Execucao da analise de redes AGN geradas e recuperadas
                //pasta para armazenar resultados da execucao.

                //chamada tipica do programa:
                //nohup ../../jvm_ibm/ibm-java2-i386-50/bin/java -jar dimreduction.jar -v ../comparacao_redes_er_concat/ 100 ER 1 false > saidaexe_comparacao_redes_100nos_er.txt &

                //parametros de entrada do programa:
                String out = args[1];//pasta para escrita dos resultados.
                int nrgenes = Integer.valueOf(args[2]); //numero de vertices da rede.
                String nettopology = args[3];//String de indetificacao da arquitetura da rede que sera gerada
                //BA==Barabasi-Albert(scale-free) //ER==Erdos-Renyi(random)//GG==Geographical Networks.
                //int concatenate = 1;//Integer.valueOf(args[4]);//concatenar sinais? 1==sim, 0==nao.
                boolean allbooleanfunctions = Boolean.valueOf(args[4]);// true para usar todas funcoes booleanas, ou false para usar conj reduzido de funcoes booleanas.

                //inicializacao das variaveis d o laco por parametros recebidos.
                int nexe = Integer.valueOf(args[5]);          //inicio do numero de execucoes para tirar a media.
                int maxexecutions = Integer.valueOf(args[6]); //MAXIMO DE EXECUCOES PARA OBTER RESULTADOS MEDIOS.
                int avgedges = Integer.valueOf(args[7]);      //inicio do grau medio
                int signalsize = Integer.valueOf(args[8]);    //inicio do tamanho do sinal
                String type_entropy = args[9];//"poor_obs";   //tipo de penalizacao aplicada. no_obs ou poor_obs
                boolean noise = Boolean.valueOf(args[10]);     //true para usar o ruido causado pela concatenacao dos sinais
                //ou false para inserir coluna de separacao entre os sinais e desconsiderar as transicoes entre concatenacoes.
                float q_entropy = 1;//Float.valueOf(args[10]);    //if selected criterion function is CoD, q_entropy = 0. Any else float value == Entropy.
                float alpha = 1;
                float beta = 0.8f;
                float thresholdentropy = 0.3f;

                int maxsignalsize = 100; //TAMANHO MAXIMO DO SINAL
                int maxconcat = 11;      //MAXIMO DE CONCATENACOES.
                int maxavgedges = 5;     //MAXIMO DE GRAU MEDIO DOS NOS.
                int sizeresultlist = 1;  //MAXIMO DE RESPOSTAS DA SELACAO DE CARACTERISTICAS.

                //parametros (default) p/ algoritmo de selecao de caracteristicas
                int quantization = 2;
                int maxfeatures = 10;

                String transitionstype = "";
                if (allbooleanfunctions) {
                    transitionstype = "all.bfs";
                } else {
                    transitionstype = "reduced.bfs";
                }
                String strnoise = "";
                if (noise) {
                    strnoise = "com.ruido";
                } else {
                    strnoise = "sem.ruido";
                }

                for (; nexe <= maxexecutions; nexe++, avgedges = 1) {
                    //geracao dos nomes dos arquivos para armazenamento
                    String nmexe = String.valueOf(nexe);
                    while (nmexe.length() < String.valueOf(maxexecutions).length()) {
                        nmexe = "0" + nmexe;
                    }
                    String res = out + "results-nexe" + nmexe + "-nrgenes-" + nrgenes + "-" + nettopology + "-" + transitionstype + "-" + strnoise + "-" + type_entropy +".txt";
                    //cabecalho do arquivo de saida.
                    IOFile.WriteFile(res, 1, null, 0, 0, 0, 0, null, 0, 1, true);
                    for (; avgedges <= maxavgedges; avgedges++, signalsize = 5) {
                        //geracao dos nomes dos arquivos para armazenamento
                        String nmavgedges = String.valueOf(avgedges);
                        String res1 = out + "original-agn-" + nettopology + "-nexe." + nmexe + "-nrnodes." + nrgenes + "-avgedges." + nmavgedges + "-" + transitionstype + ".agn";
                        File agnfile = new File(res1);
                        AGN agn = null;
                        if (agnfile.exists()) {
                            agn = IOFile.ReadAGNfromFile(res1);
                        } else {
                            agn = Topologies.CreateNetwork(
                                    nrgenes, //numero de nos da rede.
                                    signalsize, //numero de instantes de tempo.
                                    maxconcat, //numero de concatenacoes.
                                    avgedges, //media de arestas por noh.
                                    quantization, //quantizacao utilizada.
                                    nettopology, //arquitetura de rede BA ou ER.
                                    allbooleanfunctions //false == escolha entre um conjunto reduzido de funcoes booleanas;
                                    //true == todas as funcoes booleanas.
                                    );
                            IOFile.WriteAGNtoFile(agn, res1);
                        }
                        if (agn == null) {
                            System.out.println("Topologia de rede nao especificada: " + nettopology);
                            System.exit(1);
                        }

                        //armazena os valores iniciais originais
                        float[] originalinitialvalues = agn.getInitialValues();

                        for (; signalsize <= maxsignalsize; signalsize += signalsize >= 20 ? 20 : 5) {
                            //geracao dos nomes dos arquivos para armazenamento
                            String ssignalsize = String.valueOf(signalsize);
                            while (ssignalsize.length() < String.valueOf(maxsignalsize).length()) {
                                ssignalsize = "0" + ssignalsize;
                            }
                            //atribui o novo tamanho do sinal.
                            agn.setSignalsize(signalsize);
                            //atribui os mesmos valores iniciais para os resultados poderem ser comparados entre si.
                            agn.setInitialValues(originalinitialvalues);
                            //gera o sinal temporal oroginal.
                            AGNRoutines.CreateTemporalSignalq(agn);
                            int[][] generated_data = agn.getTemporalsignalquantized();

                            //debug
                            //IOFile.PrintMatrix(generated_data);

                            for (int concat = 0; concat < maxconcat; concat++) {
                                //geracao dos nomes dos arquivos para armazenamento
                                String nmconcat = String.valueOf(concat);
                                while (nmconcat.length() < String.valueOf(maxconcat).length()) {
                                    nmconcat = "0" + nmconcat;
                                }
                                float[] newinitial_values = null;
                                if (concat > 0) {
                                    newinitial_values = new float[nrgenes];
                                    //geracao aleatoria, com probabilidades iguais para os valores iniciais.
                                    //faz a geracao de um novo estado inicial.
                                    for (int i = 0; i < nrgenes; i++) {
                                        if (concat == 1) {//primeira concatenacao, todos os genes recebem valor 0 == inativos
                                            newinitial_values[i] = 0;
                                        } else if (concat == 2) {//segunda inicializacao, todos os genes recebem 1 == ativos.
                                            newinitial_values[i] = 1;
                                        } else {
                                            newinitial_values[i] = rn.nextInt(quantization);
                                        }
                                        //pode ser evitado que os genes sem preditores alterem seus valores.
                                        //if (agn.getGenes()[i].getPredictors().size() > 0){
                                        //newinitial_values[i] = rn.nextInt(quantization);
                                        //}else{
                                        //    newinitial_values[i] = agn.getGenes()[i].getValue();
                                        //}
                                    }

                                    //adicao de concatenacoes dos sinais com inicializacoes aleatorias.
                                    agn.setInitialValues(newinitial_values);
                                    AGNRoutines.CreateTemporalSignalq(agn);
                                    int[][] newinicializationdata = agn.getTemporalsignalquantized();

                                    //debug
                                    //IOFile.PrintMatrix(newinicializationdata);

                                    //concatena os dados gerados pelo estado inicial anterior e o novo estado inicial.
                                    if (noise) {//acrescentando ruido no sinal, causado pela descontinuidade da aplicacao das regras de transicao.
                                        generated_data = AGNRoutines.ConcatenateSignalq(generated_data, newinicializationdata);
                                    } else {//sem acrescentar ruido no sinal, as transicoes serao desconsideradas pelo metodo de inferencia.
                                        generated_data = AGNRoutines.ConcatenateSignalSeparatingq(generated_data, newinicializationdata);
                                    }
                                    //debug
                                    //System.out.println("\n");
                                    //IOFile.PrintMatrix(generated_data);
                                }

                                //caminho e nome que sera salvo o arquivo contendo a rede recuperada.
                                String res2 = out + "recovered-network-" + nettopology + "-nexe." + nmexe + "-nrnodes." + nrgenes + "-" + strnoise + "-avgedges." + nmavgedges + "-" + transitionstype + "-" + type_entropy + "-signalsize." + ssignalsize + "-concat." + nmconcat + ".agn";
                                AGN recoverednetwork = null;
                                File recoveredfile = new File(res2);
                                if (recoveredfile.exists()) {
                                    recoverednetwork = IOFile.ReadAGNfromFile(res2);
                                } else {
                                    recoverednetwork = new AGN(nrgenes, signalsize, quantization);
                                    recoverednetwork.setTemporalsignalquantized(generated_data);
                                    AGNRoutines.RecoverNetworkfromTemporalExpression(
                                            recoverednetwork,
                                            agn,
                                            1, //datatype: 1==temporal, 2==steady-state.
                                            false,
                                            1,//threshold_entropy
                                            type_entropy,
                                            alpha,
                                            beta,
                                            q_entropy,
                                            null, //targets
                                            maxfeatures,
                                            3,//SFFS//1==SFS, 2==Exhaustive, 3==SFFS.
                                            false,//jCB_TargetsAsPredictors.isSelected()
                                            sizeresultlist,
                                            null,
                                            "sequential",
                                            1);
                                    //armazenamento dos resultados
                                    IOFile.WriteAGNtoFile(recoverednetwork, res2);
                                }
                                //calculo da similaridade entre as redes: gerada e recuperada/identificada pelo metodo computacional.
                                System.out.println("Signal Size = " + signalsize);
                                System.out.println("Nr Concatenations = " + concat);
                                float[] CM = CNMeasurements.ConfusionMatrix(agn, recoverednetwork, thresholdentropy);
                                IOFile.WriteFile(res, 0, nettopology, nrgenes, avgedges, signalsize, quantization, CM, concat, 1, true);
                            }
                        }
                    }
                }
            } else {
                /*  EXECUCAO COM INTERFACE GRAFICA. */
                java.awt.EventQueue.invokeLater(new Runnable() {

                    @Override
                    public void run() {
                        new MainAGNWindow().setVisible(true);
                    }
                });
            }
        } else {
            /*  EXECUCAO COM INTERFACE GRAFICA. */
            java.awt.EventQueue.invokeLater(new Runnable() {

                @Override
                public void run() {
                    new MainAGNWindow().setVisible(true);
                }
            });
        }
    }
}
