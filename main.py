import random

PRINT_DEBUG = True

class IOFile:
    @staticmethod
    def read_data_first_row(filepath, startrow, startcolumn, delimiter):
        with open(filepath, 'r') as file:
            for _ in range(startrow):
                next(file)
            return file.readline().strip().split(delimiter)[startcolumn:]

    @staticmethod
    def read_data_first_column(filepath, startrow, delimiter):
        with open(filepath, 'r') as file:
            for _ in range(startrow):
                next(file)
            return [line.strip().split(delimiter)[0] for line in file]

    @staticmethod
    def read_matrix(filepath, startrow, startcolumn, delimiter):
        with open(filepath, 'r') as file:
            for _ in range(startrow):
                next(file)
            return [[float(value) for value in line.strip().split(delimiter)[startcolumn:]] for line in file]
            

    @staticmethod
    def write_matrix(filepath, matrix, delimiter):
        try:
            with open(filepath, 'w') as file:
                for row in matrix:
                    line = delimiter.join(map(str, row))
                    file.write(line + '\n')
        except IOError as error:
            raise Exception(f"Error when saving matrix: {error}")


    @staticmethod
    def write_agn_to_file(agn, filepath):
        # Placeholder for writing AGN object to file
        pass

class Preprocessing:
    SKIP_VALUE = -999

    @staticmethod
    def copy_matrix(matrix):
        return [row[:] for row in matrix]

    @staticmethod
    def normal_transform_columns(matrix, extreme_values, label):
        lines = len(matrix)
        columns = len(matrix[0])

        if extreme_values: # maybe not necessary
            means = [0] * (columns - label)
            stds = [0] * (columns - label)
        else:
            raise Exception("Error on applying normal transform.")

        for j in range(columns - label):
            if matrix[0][j] != Preprocessing.SKIP_VALUE:
                if extreme_values:
                    sum_values = sum(matrix[i][j] for i in range(lines))
                    means[j] = sum_values / lines

                    stds[j] = (sum((matrix[i][j] - means[j]) ** 2 for i in range(lines)) / (lines - 1)) ** 0.5

                if stds[j] > 0:
                    for i in range(lines):
                        matrix[i][j] = (matrix[i][j] - means[j]) / stds[j]
                else:
                    for i in range(lines):
                        matrix[i][j] = 0

    @staticmethod
    def quick_sort(arr, low, high):
        if low < high:
            pi = Preprocessing.partition(arr, low, high)
            Preprocessing.quick_sort(arr, low, pi - 1)
            Preprocessing.quick_sort(arr, pi + 1, high)

    @staticmethod
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    @staticmethod
    def quantize_columns_ma_normal(M, quantizeddata, qd, mean, std, lowthreshold, hithreshold):
        totalrows = len(M)
        totalcols = len(M[0])
        auxM = Preprocessing.copy_matrix(M)
        Preprocessing.normal_transform_columns(auxM, True, 0)

        for col in range(totalcols):
            if M[0][col] != Preprocessing.SKIP_VALUE:
                colvalues = [auxM[row][col] for row in range(totalrows)]
                mean[col] = sum(M[row][col] for row in range(totalrows)) / totalrows
                std[col] = (sum((M[row][col] - mean[col]) ** 2 for row in range(totalrows)) / (totalrows - 1)) ** 0.5

                if PRINT_DEBUG:
                    print(colvalues)
                Preprocessing.quick_sort(colvalues, 0, totalrows - 1) # right? make egual?
                if PRINT_DEBUG:
                    print(colvalues)

                negatives = [val for val in colvalues if val < 0]
                positives = [val for val in colvalues if val >= 0]
                meanneg = sum(negatives) / len(negatives) if negatives else 0
                meanpos = sum(positives) / len(positives) if positives else 0

                if std[col] == 0 or not negatives or not positives:
                    continue

                threshold = [0] * (qd - 1)
                if qd == 2:
                    index3rdq = round(0.75 * (totalrows + 1))
                    threshold[0] = colvalues[index3rdq]
                    lowthreshold[col] = threshold[0]
                    hithreshold[col] = threshold[0]
                elif qd == 3:
                    threshold[0] = meanneg
                    threshold[1] = meanpos
                    lowthreshold[col] = threshold[0]
                    hithreshold[col] = threshold[1]
                else:
                    return None

                count0 = 0
                count1 = 0
                for i in range(totalrows):
                    k = 0
                    while k < qd - 1 and auxM[i][col] > threshold[k]:
                        k += 1
                    if qd == 3:
                        if k == 2 or k == 0:
                            k = 1
                        elif k == 1:
                            k = 0
                    if k == 0:
                        count0 += 1
                    else: 
                        count1 += 1
                    quantizeddata[i][col] = k
                    
                    if PRINT_DEBUG:
                        print("Totais quantizados na coluna " + str(col) + ": " + str(count0 + count1))
                        print("zeros = " + str(count0))
                        print("ums = " + str(count1))
                        print()
            else:
                for i in range(totalrows):
                    quantizeddata[i][col] = int(M[i][col])

        return auxM

class Gene:
    def __init__(self):
        self.probsetname = None
        self.geneid = 0
        self.index = 0
        self.control = False
        self.name = ""
        self.locus = ""
        self.classe = -1
        self.elementtype = ""
        self.description = ""
        self.function = ""
        self.organism = ""
        self.chromosometype = ""
        self.chromosome = 0
        self.start = 0
        self.stop = 0
        self.synonyms = ""
        self.type = ""
        self.product = ""
        self.proteinid = ""
        self.pathway = []
        self.pathwaydescription = []
        self.x = 0.0
        self.y = 0.0
        self.value = 0.0
        self.predictors = []
        self.targets = []
        self.booleanfunctions = []
        self.cfvalues = []
        self.predictorsties = []
        self.probtable = []

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def get_predictors(self):
        return self.predictors

    def set_predictors(self, predictors):
        self.predictors = predictors

    def add_predictor(self, predictor, cfvalue=None):
        self.predictors.append(predictor)
        if cfvalue is not None:
            self.cfvalues.append(cfvalue)

    def remove_predictor(self, predictor):
        self.predictors = [p for p in self.predictors if p != predictor]

    def remove_target(self, target):
        self.targets = [t for t in self.targets if t != target]

    def get_targets(self):
        return self.targets

    def set_targets(self, targets):
        self.targets = targets

    def add_target(self, target):
        self.targets.append(target)

    def get_cfvalues(self):
        return self.cfvalues

    def set_cfvalues(self, cfvalues):
        self.cfvalues = cfvalues

    def set_cfvalue(self, cfvalue, position=None):
        if position is not None and position >= 0:
            self.cfvalues.insert(position, cfvalue)
        else:
            self.cfvalues.append(cfvalue)

    def get_description(self):
        return self.description

    def set_description(self, description):
        self.description = description

    def get_booleanfunctions(self):
        return self.booleanfunctions

    def set_booleanfunctions(self, booleanfunctions):
        self.booleanfunctions = booleanfunctions

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_locus(self):
        return self.locus

    def set_locus(self, locus):
        self.locus = locus

    def get_organism(self):
        return self.organism

    def set_organism(self, organism):
        self.organism = organism

    def get_classe(self):
        return self.classe

    def set_classe(self, classe):
        self.classe = classe

    def get_elementtype(self):
        return self.elementtype

    def set_elementtype(self, elementtype):
        self.elementtype = elementtype

    def get_start(self):
        return self.start

    def set_start(self, start):
        self.start = start

    def get_stop(self):
        return self.stop

    def set_stop(self, stop):
        self.stop = stop

    def get_synonyms(self):
        return self.synonyms

    def set_synonyms(self, synonyms):
        self.synonyms = synonyms

    def get_function(self):
        return self.function

    def set_function(self, function):
        self.function = function

    def get_index(self):
        return self.index

    def set_index(self, index):
        self.index = index

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type

    def get_probsetname(self):
        return self.probsetname

    def set_probsetname(self, probsetname):
        self.probsetname = probsetname

    def get_geneid(self):
        return self.geneid

    def set_geneid(self, geneid):
        self.geneid = geneid

    def get_product(self):
        return self.product

    def set_product(self, product):
        self.product = product

    def get_proteinid(self):
        return self.proteinid

    def set_proteinid(self, proteinid):
        self.proteinid = proteinid

    def get_chromosome(self):
        return self.chromosome

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome

    def get_predictorsties(self):
        return self.predictorsties

    def set_predictorsties(self, predictorsties):
        self.predictorsties = predictorsties

    def is_control(self):
        return self.control

    def set_control(self, control):
        self.control = control

    def get_chromosometype(self):
        return self.chromosometype

    def set_chromosometype(self, chromosometype):
        self.chromosometype = chromosometype

    def get_probtable(self):
        return self.probtable

    def set_probtable(self, probtable):
        self.probtable = probtable

    def get_pathway(self):
        return self.pathway

    def set_pathway(self, pathway):
        self.pathway = pathway

    def get_pathwaydescription(self):
        return self.pathwaydescription

    def set_pathwaydescription(self, pathwaydescription):
        self.pathwaydescription = pathwaydescription

class AGN:
    def __init__(self, nrgenes, signalsize, quantization, topology=None, nrinitializations=0, avgedges=0.0, allbooleanfunctions=False):
        self.topology = topology
        self.nrgenes = nrgenes
        self.signalsize = signalsize
        self.nrinitializations = nrinitializations
        self.quantization = quantization
        self.avgedges = avgedges
        self.allbooleanfunctions = allbooleanfunctions
        self.genes = [Gene() for _ in range(nrgenes)]
        self.temporalsignal = None
        self.temporalsignalnormalized = None
        self.temporalsignalquantized = None
        self.labelstemporalsignal = None
        self.mean = None
        self.std = None
        self.lowthreshold = None
        self.hithreshold = None
        self.removedgenes = None

        rn = random.Random()
        rn.seed()

        for i in range(nrgenes):
            self.genes[i].set_name(f"g{i}")
            self.genes[i].set_probsetname(f"g{i}")
            self.genes[i].set_description(f"g{i}")
            self.genes[i].set_index(i)
            self.genes[i].set_value(float(rn.randint(0, quantization)))

    def get_initial_values(self):
        return [gene.get_value() for gene in self.genes]

    def set_initial_values(self, initialvalues):
        for i in range(self.nrgenes):
            self.genes[i].set_value(initialvalues[i])

    def get_topology(self):
        return self.topology

    def set_topology(self, topology):
        self.topology = topology

    def get_nrgenes(self):
        return self.nrgenes

    def set_nrgenes(self, nrgenes):
        self.nrgenes = nrgenes

    def get_signalsize(self):
        return self.signalsize

    def set_signalsize(self, signalsize):
        self.signalsize = signalsize

    def get_quantization(self):
        return self.quantization

    def set_quantization(self, quantization):
        self.quantization = quantization

    def get_avgedges(self):
        return self.avgedges

    def set_avgedges(self, avgedges):
        self.avgedges = avgedges

    def is_allbooleanfunctions(self):
        return self.allbooleanfunctions

    def set_allbooleanfunctions(self, allbooleanfunctions):
        self.allbooleanfunctions = allbooleanfunctions

    def get_nrinitializations(self):
        return self.nrinitializations

    def set_nrinitializations(self, nrinitializations):
        self.nrinitializations = nrinitializations

    def get_labelstemporalsignal(self):
        return self.labelstemporalsignal

    def set_labelstemporalsignal(self, labelstemporalsignal):
        self.labelstemporalsignal = labelstemporalsignal

    def get_genes(self):
        return self.genes

    def set_genes(self, genes):
        self.genes = genes

    def get_temporalsignal(self):
        return self.temporalsignal

    def set_temporalsignal(self, temporalsignal):
        self.temporalsignal = temporalsignal

    def set_temporal_signal(self, temporalsignal, labels):
        self.set_temporalsignal(temporalsignal)
        self.set_labelstemporalsignal(labels)

    def get_mean(self):
        return self.mean

    def set_mean(self, mean):
        self.mean = mean

    def get_std(self):
        return self.std

    def set_std(self, std):
        self.std = std

    def get_removedgenes(self):
        return self.removedgenes

    def set_removedgenes(self, removedgenes):
        self.removedgenes = removedgenes

    def get_lowthreshold(self):
        return self.lowthreshold

    def set_lowthreshold(self, lowthreshold):
        self.lowthreshold = lowthreshold

    def get_hithreshold(self):
        return self.hithreshold

    def set_hithreshold(self, hithreshold):
        self.hithreshold = hithreshold

    def get_temporalsignalnormalized(self):
        return self.temporalsignalnormalized

    def set_temporalsignalnormalized(self, temporalsignalnormalized):
        self.temporalsignalnormalized = temporalsignalnormalized

    def get_temporalsignalquantized(self):
        return self.temporalsignalquantized

    def set_temporalsignalquantized(self, temporalsignalquantized):
        self.temporalsignalquantized = temporalsignalquantized

    def set_temporalsignalquantized_from_float(self, temporalsignalquantized):
        self.temporalsignalquantized = [[int(value) for value in row] for row in temporalsignalquantized]

class AGNRoutines:
    @staticmethod
    def set_gene_names(agn, genenames):
        if agn.get_nrgenes() == len(genenames):
            for g in range(len(agn.get_genes())):
                name = genenames[g]
                agn.get_genes()[g].set_name(name)
        else:
            print("Error on labeling genes, size does not match.")

    @staticmethod
    def recover_network_from_temporal_expression(agn, datatype, threshold_entropy, type_entropy, alpha, beta, q_entropy, maxfeatures, resultsetsize):
        # Placeholder for network recovery logic
        pass

def main():
    delimiter = "\t"
    inpath = "./data/QCDataSet-first300-with-labels.txt"
    outpath = "./out"

    threshold_entropy = 1
    type_entropy = "no_obs"
    alpha = 1
    q_entropy = 1
    beta = 0.8
    maxfeatures = 3
    resultsetsize = 1


    featurestitles = IOFile.read_data_first_row(inpath, 0, 1, delimiter)
    startrow = 1
    genenames = IOFile.read_data_first_column(inpath, startrow, delimiter)
    startcolumn = 1
    expressiondata = IOFile.read_matrix(inpath, startrow, startcolumn, delimiter)
    
    nrgenes = len(expressiondata) # or len(genenames)
    signalsize = len(expressiondata[0]) # or len(featurestitles)

    mean = [0] * signalsize
    std = [0] * signalsize
    lowthreshold = [0] * signalsize
    hithreshold = [0] * signalsize
    quantizeddata = [[0] * signalsize for _ in range(nrgenes)]

    normalizeddata = Preprocessing.quantize_columns_ma_normal(
        expressiondata,
        quantizeddata,
        2,
        mean,
        std,
        lowthreshold,
        hithreshold
    )

    IOFile.write_matrix(outpath + "normalized-log2.txt", normalizeddata, ";")
    IOFile.write_matrix(outpath + "quantized-log2.txt", quantizeddata, ";")

    recoverednetwork = AGN(nrgenes, signalsize, 2)
    recoverednetwork.set_mean(mean)
    recoverednetwork.set_std(std)
    recoverednetwork.set_lowthreshold(lowthreshold)
    recoverednetwork.set_hithreshold(hithreshold)
    recoverednetwork.set_temporalsignal(expressiondata)
    recoverednetwork.set_temporalsignalquantized(quantizeddata)
    recoverednetwork.set_temporalsignalnormalized(normalizeddata)
    recoverednetwork.set_labelstemporalsignal(featurestitles)
    AGNRoutines.set_gene_names(recoverednetwork, genenames)

    # AGNRoutines.recover_network_from_temporal_expression(
    #     recoverednetwork,
    #     1,  # datatype: 1==temporal, 2==steady-state.
    #     threshold_entropy,
    #     type_entropy,
    #     alpha,
    #     beta,
    #     q_entropy,
    #     maxfeatures,
    #     resultsetsize
    # )

    # IOFile.write_agn_to_file(recoverednetwork, outpath + "log2-complete.agn")

if __name__ == "__main__":
    main()