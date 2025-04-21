import random
import math
import pickle
import os
from dotenv import load_dotenv
import threading
from datetime import datetime
import time
import gc
import concurrent.futures

def sum_of_squares(n):
    print(f"sum_of_squares({n})")
    return sum(i * i for i in range(n))


PRINT_DEBUG = True
VERBOSE_LEVEL = {
    "NONE": 0,
    "ERROR": 1,
    "WARNING": 2,
    "INFO": 3,
    "TIMER": 4,
    "DEBUG": 5
}

def chr_4_digit(int_value):
    # return f"\\u{(int_value):04x}"
    return chr(int_value)

def ord_4_digit(unicode_escape):
    # return int(unicode_escape[2:], 16)
    return ord(unicode_escape)

# int or ord function
def int_or_ord_4_digit(value):
    # # return ord_4_digit(value)
    # if isinstance(value, str) and value.startswith("\\u"):
    #     return ord_4_digit(value)
    # try:
    #     return int(value)
    # except ValueError:
    #     # try:
    #     return int(float(value))
    #     # except ValueError:
    #     #     return ord_4_digit(value)
    try:
        return ord_4_digit(value)
    except TypeError:
        try:
            return int(value)
        except ValueError:
            return int(float(value))

class MathRoutines:
    @staticmethod
    def number_combinations(n, c):
        combinations = 1
        for i in range(n, n - c, -1):
            combinations *= i
        for i in range(2, c + 1):
            combinations //= i
        return combinations

    @staticmethod
    def get_column_average(M, column, classe=None):
        sum_values = 0
        count = 0
        for row in M:
            if classe is None or int(row[-1]) == classe:
                sum_values += row[column]
                count += 1
        return sum_values / count if count > 0 else 0

    @staticmethod
    def get_std(M, average, column, classe):
        sum_values = 0
        for row in M:
            if int(row[-1]) == classe:
                sum_values += (row[column] - average) ** 2
        return math.sqrt(sum_values)

    @staticmethod
    def get_correlation_coeficient_classes(Md, features):
        lines = len(Md)
        columns = len(Md[0])
        classes = max(row[-1] for row in Md) + 1
        avg_values = [[MathRoutines.get_column_average(Md, f, c) for f in features] for c in range(classes)]
        avg_classes = [sum(avg_values[c]) / len(features) for c in range(classes)]
        std_values = [math.sqrt(sum((avg_values[c][f] - avg_classes[c]) ** 2 for f in range(len(features))) / len(features)) for c in range(classes)]
        output = [[0] * classes for _ in range(classes)]
        for c1 in range(classes):
            for c2 in range(c1 + 1, classes):
                sum_c1c2 = sum((avg_values[c1][f] - avg_classes[c1]) * (avg_values[c2][f] - avg_classes[c2]) for f in range(len(features)))
                output[c1][c2] = sum_c1c2 / (std_values[c1] * std_values[c2])
        return output

    @staticmethod
    def get_correlation_coeficient_features(Md, features):
        avg_values = [MathRoutines.get_column_average(Md, f) for f in features]
        std_values = [math.sqrt(sum((row[f] - avg_values[i]) ** 2 for row in Md) / len(Md)) for i, f in enumerate(features)]
        output = [[0] * len(features) for _ in range(len(features))]
        for f1 in range(len(features)):
            for f2 in range(f1 + 1, len(features)):
                sum_f1f2 = sum((row[features[f1]] - avg_values[f1]) * (row[features[f2]] - avg_values[f2]) for row in Md)
                output[f1][f2] = sum_f1f2 / (std_values[f1] * std_values[f2])
        return output

    @staticmethod
    def transpose_matrix(M):
        return list(map(list, zip(*M)))

    @staticmethod
    def bin2dec(bin_str):
        dec = 0
        size = len(bin_str) - 1
        or_ = 1
        for i in range(size + 1):
            vs = bin_str[size - i]
            v = int(vs)
            dec += v * or_
            if or_ == 1:
                or_ = 2
            else:
                or_ *= 2
                
        return dec
        

    @staticmethod
    def base_n2dec(str_val, base):
        dec = 0
        size = len(str_val) - 1
        or_ = 1
        for i in range(size + 1):
            vs = str_val[size - i]
            v = int(vs)
            dec += v * or_
            if or_ == 1:
                or_ = base
            else:
                or_ *= base
                
        return dec

    @staticmethod
    def dec2base_n(dec, base, size):
        result = []
        while dec > 0:
            result.append(str(dec % base))
            dec //= base
        while len(result) < size:
            result.append('0')
        return ''.join(reversed(result))

    @staticmethod
    def str2boolean(bindigits):
        return [char == '1' for char in bindigits]

    @staticmethod
    def int2boolean(bindigit):
        return bindigit == 1

    @staticmethod
    def boolean2int(bindigit):
        return 1 if bindigit else 0

    @staticmethod
    def float2char(M):
        lines = len(M)
        columns = len(M[0])
        M2 = [[''] * columns for _ in range(lines)]
        for i in range(lines):
            for j in range(columns):
                M2[i][j] = chr_4_digit(int(M[i][j]))
        return M2

class IOFile:
    VERBOSITY_LEVEL = VERBOSE_LEVEL["DEBUG"]  # Default verbosity level
    @staticmethod
    def set_verbosity(level):
        IOFile.VERBOSITY_LEVEL = level

    @staticmethod
    def write_ties(originalagn, tiesout, targetindex, avgedges, topology, originalpredictors, q_entropy, predictors, ties, h_global, flag):
        pass
    
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


    def write_agn_to_file(self, agn, filepath):
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(agn, file)
            return True
        except IOError as e:
            self.print_and_log(f"Error when creating FileOutputStream: {e}", verbosity=VERBOSE_LEVEL["ERROR"])
            return False
        
    @staticmethod
    def print_and_log(*message, end="\n", path="logs/logs.log", verbosity=VERBOSE_LEVEL["DEBUG"]):
        if verbosity <= IOFile.VERBOSITY_LEVEL:
            print(" ".join(map(str, message)), end=end)
            with open(path, 'a') as file:
                file.write(f"{' '.join(map(str, message))}{end}")

class Timer:
    PATH = "timing/timers.log"
    VERBOSITY = VERBOSE_LEVEL["DEBUG"]
    
    def __init__(self, verbosity=VERBOSE_LEVEL["DEBUG"]):
        self.timers = {}
        self.VERBOSITY = verbosity

    def set_verbosity(self, level):
        self.VERBOSITY = level
    
    def print_and_log(self, message):
        IOFile.print_and_log(message, path=self.PATH, verbosity=VERBOSE_LEVEL["TIMER"])
    
    def start(self, name):
        if True:
            return
        if VERBOSE_LEVEL["TIMER"] < self.VERBOSITY:
            return
        if name in self.timers:
            self.print_and_log(f"Timer {name} is already running.")
        else:
            self.timers[name] = time.time()
            self.print_and_log(f"Timer {name} started at {datetime.now()}")

    def end(self, name):
        if True:
            return
        if VERBOSE_LEVEL["TIMER"] < self.VERBOSITY:
            return
        if name not in self.timers:
            self.print_and_log(f"Timer {name} was not started.")
        else:
            start_time = self.timers.pop(name)
            end_time = time.time()
            duration = end_time - start_time
            self.print_and_log(f"Timer {name} ended at {datetime.now()}")
            self.print_and_log(f"Duration for {name}: {duration:.4f} seconds")

class Preprocessing:
    SKIP_VALUE = -999

    @staticmethod
    def max_min_column(M, col):
        mm = [M[0][col], M[0][col]]
        for row in M:
            if row[col] > mm[0]:
                mm[0] = row[col]
            if row[col] < mm[1]:
                mm[1] = row[col]
        return mm

    @staticmethod
    def max_min_row(M, row, label):
        mm = [M[row][0], M[row][0]]
        for j in range(len(M[0]) - label):
            if M[row][j] > mm[0]:
                mm[0] = M[row][j]
            if M[row][j] < mm[1]:
                mm[1] = M[row][j]
        return mm

    @staticmethod
    def max_min(M):
        mm = [M[0][0], M[0][0]]
        for row in M:
            for value in row:
                if value > mm[0]:
                    mm[0] = value
                if value < mm[1]:
                    mm[1] = value
        return mm

    @staticmethod
    def filter_ma(expressiondata, geneids):
        remaingenes = []
        removedgenes = []
        for lin in range(len(expressiondata)):
            if all(value != 0 for value in expressiondata[lin]):
                remaingenes.append(lin)
            else:
                removedgenes.append(geneids[0][lin])
                IOFile.print_and_log(f"Gene {geneids[0][lin]} was removed by filter.")
        IOFile.print_and_log(f"{len(removedgenes)} removed genes.")
        filtereddata = [[expressiondata[lin][col] for col in range(len(expressiondata[0]))] for lin in remaingenes]
        return filtereddata

    @staticmethod
    def apply_log2(expressiondata):
        filtereddata = [[((math.log(value) / math.log(2)) if value != Preprocessing.SKIP_VALUE else value) for value in row] for row in expressiondata]
        return filtereddata

    @staticmethod
    def copy_matrix(matrix):
        return [row[:] for row in matrix]

    @staticmethod
    def normal_transform_columns(matrix, extreme_values, label):
        lines = len(matrix)
        columns = len(matrix[0])

        if extreme_values: # maybe not necessary
            means = [0.0] * (columns - label)
            stds = [0.0] * (columns - label)
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
    def selection_sort(M):
        for j in range(len(M) - 1):
            minvalue = M[j][0]
            minposition = j
            for i in range(j + 1, len(M)):
                if M[i][0] < minvalue:
                    minvalue = M[i][0]
                    minposition = i
            if minposition != j:
                M[j], M[minposition] = M[minposition], M[j]

    @staticmethod
    def bubble_sort(M):
        change = True
        while change:
            change = False
            for i in range(len(M) - 1):
                if M[i][0] > M[i + 1][0]:
                    M[i], M[i + 1] = M[i + 1], M[i]
                    change = True

    @staticmethod
    def normalize(M, qd, label):
        for lin in range(len(M)):
            maxmin = Preprocessing.max_min(M[lin])
            for col in range(len(M[lin]) - label):
                normalizedvalue = (qd - 1) * (M[lin][col] - maxmin[1]) / (maxmin[0] - maxmin[1])
                k = 0
                for k in range(qd - 1):
                    if normalizedvalue <= k:
                        break
                M[lin][col] = k

    @staticmethod
    def quantize_rows(M, qd, extreme_values, label):
        lines = len(M)
        columns = len(M[0])
        Preprocessing.normal_transform_lines(M, extreme_values, label)

        for j in range(lines):
            if M[0][j] != Preprocessing.SKIP_VALUE:
                negatives = []
                positives = []
                for i in range(columns - label):
                    if M[j][i] < 0:
                        negatives.append(M[j][i])
                    else:
                        positives.append(M[j][i])
                meanneg = sum(negatives) / len(negatives) if negatives else 0
                meanpos = sum(positives) / len(positives) if positives else 0


                ind_threshold = 0
                increment = -meanneg / (qd / 2)
                threshold = [0.0] * (qd - 1)
                i = meanneg + increment
                while i < 0:
                    threshold[ind_threshold] = i
                    ind_threshold += 1
                    i += increment
                increment = meanpos / (qd / 2)
                ind_threshold = qd - 2
                i = meanpos - increment
                while i > 0:
                    threshold[ind_threshold] = i
                    ind_threshold -= 1
                    i -= increment

                for i in range(columns - label):
                    k = 0
                    while k < qd - 1:
                        if threshold[k] >= M[j][i]:
                            break
                        k += 1
                    M[j][i] = k

    @staticmethod
    def quantize_columns(M, qd, extreme_values, label):
        Preprocessing.normal_transform_columns(M, extreme_values, label)
        for j in range(len(M[0]) - label):
            negatives = []
            positives = []
            for i in range(len(M)):
                if M[i][j] < 0:
                    negatives.append(M[i][j])
                else:
                    positives.append(M[i][j])
            meanneg = sum(negatives) / len(negatives) if negatives else 0
            meanpos = sum(positives) / len(positives) if positives else 0
            # // obtaining the thresholds for quantization
            #     int indThreshold = 0;
            #     double increment = -meanneg / ((double) qd / 2);
            #     double[] threshold = new double[qd - 1];
            #     for (double i = meanneg + increment; i < 0; i += increment, indThreshold++) {
            #         threshold[indThreshold] = i;
            #     }
            #     increment = meanpos / ((double) qd / 2);
            #     indThreshold = qd - 2;
            #     for (double i = meanpos - increment; i > 0; i -= increment, indThreshold--) {
            #         threshold[indThreshold] = i;
            #         // quantizing the feature values
            #     }
            ind_threshold = 0
            increment = -meanneg / (qd / 2)
            threshold = [0] * (qd - 1)
            i = meanneg + increment
            while i < 0:
                threshold[ind_threshold] = i
                ind_threshold += 1
                i += increment
            increment = meanpos / (qd / 2)
            ind_threshold = qd - 2
            i = meanpos - increment
            while i > 0:
                threshold[ind_threshold] = i
                ind_threshold -= 1
                i -= increment
            
            for i in range(len(M)):
                k = 0
                while k < qd - 1:
                    if threshold[k] >= M[i][j]:
                        break
                    k += 1
                M[i][j] = k

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
                    IOFile.print_and_log(colvalues)
                Preprocessing.quick_sort(colvalues, 0, totalrows - 1) # right? make egual?
                if PRINT_DEBUG:
                    IOFile.print_and_log(colvalues)

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
                        IOFile.print_and_log("Totais quantizados na coluna " + str(col) + ": " + str(count0 + count1))
                        IOFile.print_and_log("zeros = " + str(count0))
                        IOFile.print_and_log("ums = " + str(count1))
                        IOFile.print_and_log()
            else:
                for i in range(totalrows):
                    quantizeddata[i][col] = int(M[i][col])

        return auxM

    @staticmethod
    def normal_transform_lines(M, extreme_values, label):
        lines = len(M)
        columns = len(M[0])

        if extreme_values:
            means = [0.0] * lines
            stds = [0.0] * lines
        else:
            raise Exception("Error on applying normal transform.")

        for j in range(lines):
            if extreme_values:
                sum_values = sum(M[j][i] for i in range(columns - label))
                means[j] = sum_values / (columns - label)

                stds[j] = sum((M[j][i] - means[j]) ** 2 for i in range(columns - label))
                stds[j] /= (columns - label - 1)
                stds[j] = math.sqrt(stds[j])

            if stds[j] > 0:
                for i in range(columns - label):
                    M[j][i] -= means[j]
                    M[j][i] /= stds[j]
            else:
                for i in range(columns - label):
                    M[j][i] = 0

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
            self.genes[i].set_value(rn.randint(0, quantization))

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

class FS:
    def __init__(self, samples, npv, nc, typeMCE_COD, alphaPenalty, betaConfidence, qentropy, maxresultlistsize):
        self.I = []
        self.probtable = []
        self.columns = len(samples[0])
        self.h_global = 1.0
        self.A = samples
        self.n = npv
        self.c = nc
        self.type = typeMCE_COD
        self.alpha = alphaPenalty
        self.beta = betaConfidence
        self.q = qentropy
        self.itmax = int(math.floor(math.log(len(samples)) / math.log(npv)))
        self.resultlist = []
        self.resultlistsize = maxresultlistsize
        self.maxresultvalue = 1
        self.bestentropy = []
        self.bestset = []
        self.tiesentropy = []
        self.ties = []
        self.jointentropiesties = []
        self.timer = Timer(IOFile.VERBOSITY_LEVEL)

    def insert_in_result_list(self, I, hmin):
        item = [hmin, I[:]]
        if len(self.resultlist) < self.resultlistsize:
            self.resultlist.append(item)
            if len(self.resultlist) > 1:
                Preprocessing.selection_sort(self.resultlist)
        else:
            vi = item[0]
            vs = self.resultlist[-1][0]
            if vi < vs:
                self.resultlist[-1] = item
                Preprocessing.selection_sort(self.resultlist)

    def break_ties(self, i):
        if not self.ties[i] or self.tiesentropy[i] == 1:
            return
        self.jointentropiesties = [0] * len(self.ties[i])
        maxjointentropy = 0
        maxjointentropyposition = 0
        for p in range(len(self.ties[i])):
            predictors = self.ties[i][p]
            self.jointentropiesties[p] = Criteria.joint_entropy(self.n, predictors, self.A, self.c)
            if self.jointentropiesties[p] > maxjointentropy:
                maxjointentropy = self.jointentropiesties[p]
                maxjointentropyposition = p
        self.I = self.ties[i][maxjointentropyposition]

    def minimal(self, maxsetsize):
        posminimal = 0
        for i in range(1, maxsetsize + 1):
            if self.bestentropy[i] < self.h_global:
                self.h_global = self.bestentropy[i]
                self.I = self.bestset[i]
                posminimal = i
        if self.ties[posminimal] and len(self.ties[posminimal]) > 1:
            self.break_ties(posminimal)
        cfvalue, probtable = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, self.I, self.A, self.q)
        self.probtable = probtable[:]

    def minimal_ma(self, maxsetsize):
        posminimal = 0
        for i in range(1, maxsetsize + 1):
            if self.bestentropy[i] <= self.h_global:
                self.h_global = self.bestentropy[i]
                self.I = self.bestset[i]
                posminimal = i
        if self.ties[posminimal] and len(self.ties[posminimal]) > 1:
            self.break_ties(posminimal)
        cfvalue, probtable = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, self.I, self.A, self.q)
        self.probtable = probtable[:]

    def inicialize(self, maxfeatures):
        self.bestentropy = [1] * maxfeatures
        self.bestset = [[] for _ in range(maxfeatures)]
        self.tiesentropy = [1] * maxfeatures
        self.ties = [[] for _ in range(maxfeatures)]

    def run_sfs(self, called_by_exhaustive, maxfeatures):
        columns = len(self.A[0])
        for i in range(min(columns - 1, maxfeatures)):
            h_min = 1.1
            f_min = -1
            H = 1
            self.I.append(-1)
            for f in range(columns - 1):
                if f in self.I:
                    continue
                self.I[-1] = f
                # self.timer.start("MCE_COD")
                H, probtable = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, self.I[:], self.A[:], self.q)
                # self.timer.end("MCE_COD")
                if H < h_min:
                    f_min = f
                    h_min = H
                    self.insert_in_result_list(self.I, H)
                if H == 0:
                    break
            if h_min < self.h_global:
                self.I[-1] = f_min
                self.h_global = h_min
                if self.h_global == 0:
                    break
            else:
                self.I.pop()
                break
        if called_by_exhaustive:
            self.itmax = len(self.I)

    def best_set(self, bestset, bestentropy, other, entropy):
        size = len(other)
        if entropy < bestentropy[size]:
            bestentropy[size] = entropy
            bestset.clear()
            bestset.extend(other)

    def run_sffs(self, maxfeatures, targetindex, agn):
        if maxfeatures >= self.columns:
            maxfeatures = self.columns - 1
        self.inicialize(maxfeatures + 1)
        while len(self.I) < maxfeatures:
            h_min = 1
            f_min = -1
            H = 1
            self.I.append(-1)
            for f in range(self.columns - 1):
                if agn:
                    predictorindex = f
                    if predictorindex >= targetindex:
                        predictorindex += 1
                    if agn.get_genes()[predictorindex].is_control():
                        continue
                if f in self.I:
                    continue
                self.I[-1] = f
                H, current_probtable = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, self.I, self.A, self.q)
                if H < h_min:
                    f_min = f
                    h_min = H
                    self.insert_in_result_list(self.I, H)
                if abs(H - h_min) < 0.00001:
                    if not self.ties[len(self.I)]:
                        self.ties[len(self.I)] = []
                    if H < self.tiesentropy[len(self.I)]:
                        self.ties[len(self.I)].clear()
                        self.tiesentropy[len(self.I)] = H
                    titem = self.I[:]
                    self.ties[len(self.I)].append(titem)
                    self.probtable = current_probtable  # Store in the FS instance
            if len(self.I) <= maxfeatures and f_min != -1:
                self.I[-1] = f_min
                if not self.bestset[len(self.I)]:
                    self.bestset[len(self.I)] = self.I[:]
                    self.bestentropy[len(self.I)] = h_min
                else:
                    self.best_set(self.bestset[len(self.I)], self.bestentropy, self.I, h_min)
                repeticoes = 0
                for be in range(1, len(self.bestentropy) - 1):
                    if self.bestentropy[be] < 1 and abs(self.bestentropy[be] - self.bestentropy[be + 1]) < 0.001:
                        repeticoes += 1
                if repeticoes > 1:
                    break
                again = True
                while len(self.I) > 2 and again:
                    combinations = len(self.I)
                    le = 1
                    lmf = -1
                    for comb in range(combinations):
                        xk = [self.I[nc] for nc in range(combinations) if nc != comb]
                        nh = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, xk, self.A, self.q)
                        if nh < le:
                            le = nh
                            lmf = self.I[comb]
                    if le < self.bestentropy[len(self.I) - 1] and lmf != f_min:
                        self.I.remove(lmf)
                        self.best_set(self.bestset[len(self.I)], self.bestentropy, self.I, le)
                        self.insert_in_result_list(self.I, le)
                        again = True
                        f_min = -1
                        self.ties[len(self.I)].clear()
                        self.tiesentropy[len(self.I)] = le
                        titem = self.I[:]
                        self.ties[len(self.I)].append(titem)
                    else:
                        again = False
                if h_min == 0:
                    break
            else:
                self.I.pop()
                break
        self.minimal(maxfeatures)

    def run_sffs_stack(self, maxfeatures, targetindex, agn):
        IOFile.print_and_log(f"Running Target index == {targetindex}")
        if agn:
            IOFile.print_and_log(f", name == {agn.get_genes()[targetindex].get_name()}")
        IOFile.print_and_log("\n")
        if maxfeatures >= self.columns:
            maxfeatures = self.columns - 1
        self.inicialize(maxfeatures + 1)
        exestack = [[-1]]
        expandedestack = []
        while exestack:
            h_min = 1
            f_min = -1
            H = 1
            self.I = exestack.pop(0)
            expandedestack.append(self.I[:])
            IOFile.print_and_log("\nExpanded tied predictors: ", self.I)
            for f in range(self.columns - 1):
                if agn:
                    predictorindex = f
                    if predictorindex >= targetindex:
                        predictorindex += 1
                    if agn.get_genes()[predictorindex].is_control():
                        continue
                if f in self.I:
                    continue
                self.I[-1] = f
                H, current_probtable = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, self.I, self.A, self.q)
                if H < h_min:
                    f_min = f
                    h_min = H
                    self.insert_in_result_list(self.I, H)
                if abs(H - h_min) < 0.001:
                    if H < self.tiesentropy[len(self.I)]:
                        self.ties[len(self.I)].clear()
                        self.tiesentropy[len(self.I)] = H
                    if abs(H - self.tiesentropy[len(self.I)]) < 0.001:
                        titem = self.I[:]
                        if not self.contain_predictor_set(self.ties[len(self.I)], titem):
                            self.ties[len(self.I)].append(titem)
            if len(self.I) <= maxfeatures and f_min != -1:
                self.I[-1] = f_min
                self.best_set(self.bestset[len(self.I)], self.bestentropy, self.I, h_min)
                again = True
                while len(self.I) > 2 and again:
                    combinations = len(self.I)
                    le = 1
                    lmf = -1
                    for comb in range(combinations):
                        xk = [self.I[nc] for nc in range(combinations) if nc != comb]
                        nh = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, xk, self.A, self.q)
                        if nh < le:
                            le = nh
                            lmf = self.I[comb]
                    if le < self.bestentropy[len(self.I) - 1] and lmf != f_min:
                        self.I.remove(lmf)
                        self.best_set(self.bestset[len(self.I)], self.bestentropy, self.I, le)
                        self.insert_in_result_list(self.I, le)
                        again = True
                        f_min = -1
                        self.ties[len(self.I)].clear()
                        self.tiesentropy[len(self.I)] = le
                        titem = self.I[:]
                        self.ties[len(self.I)].append(titem)
                    else:
                        again = False
                IOFile.print_and_log(f"Preditores escolhidos com cardinalidade == {len(self.I)}")
                IOFile.print_and_log(f"Preditores escolhidos: {self.I}")
                IOFile.print_and_log("Preditores empatados empilhados:")
                contp = 0
                for t in range(len(self.ties[len(self.I)])):
                    predictorset = self.ties[len(self.I)][t][:]
                    predictorset.append(-1)
                    if not self.contain_predictor_set(exestack, predictorset) and not self.contain_predictor_set(expandedestack, predictorset) and len(predictorset) <= maxfeatures:
                        exestack.append(predictorset)
                        IOFile.print_and_log(predictorset)
                        contp += 1
                IOFile.print_and_log(f"# empilhados == {contp}")
                IOFile.print_and_log(f"Tamanho da pilha == {len(exestack)}")
            else:
                self.I.pop()
                break
        IOFile.print_and_log(f"Numero de conjuntos de preditores expandidos == {len(expandedestack)}")
        self.minimal_ma(maxfeatures)

    def contain_predictor_set(self, stack, predictorset):
        for stackset in stack:
            count = 0
            for predictor in predictorset:
                if predictor in stackset:
                    count += 1
            if count == len(predictorset):
                return True
        return False

    def run_exhaustive(self, it, f, tempI):
        if self.itmax == 1:
            for i in range(self.columns - 1):
                tempI.append(i)
                H, current_probtable = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, tempI, self.A, self.q)
                if H < self.h_global:
                    self.I = tempI[:]
                    self.h_global = H
                    self.insert_in_result_list(tempI, H)
                    self.probtable = current_probtable  # Store in the FS instance
                tempI.pop()
            return
        tempI.append(f)
        if it >= self.itmax - 1:
            H, current_probtable = Criteria.MCE_COD(self.type, self.alpha, self.beta, self.n, self.c, tempI, self.A, self.q)
            if H < self.h_global:
                self.I = tempI[:]
                self.h_global = H
                self.insert_in_result_list(tempI, H)
                self.probtable = current_probtable  # Store in the FS instance
            return
        for i in range(f + 1, self.columns - 1):
            self.run_exhaustive(it + 1, i, tempI)
            tempI.pop()
        if it == 0 and f < self.columns - self.itmax:
            tempI.clear()
            self.run_exhaustive(0, f + 1, tempI)


class FIFOQueue:
    DEFAULT_SIZE = 10

    def __init__(self, size=DEFAULT_SIZE):
        self.m_values = [None] * size
        self.clear()

    def enq(self, element):
        if self.m_full:
            raise Exception("Queue full.")
        self.m_empty = False

        self.m_values[self.m_in] = element
        self.m_in += 1
        if self.m_in == len(self.m_values):
            self.m_in = 0
        self.m_full = self.m_in == self.m_out

    def deq(self):
        if self.m_empty:
            raise Exception("Queue empty.")
        self.m_full = False

        o = self.m_values[self.m_out]
        self.m_values[self.m_out] = None
        self.m_out += 1

        if self.m_out == len(self.m_values):
            self.m_out = 0
        self.m_empty = self.m_in == self.m_out

        return o

    def size(self):
        if self.m_empty:
            return 0
        if self.m_full:
            return len(self.m_values)

        size = 0
        for i in range(self.m_out, len(self.m_values)):
            size += 1
            if i == self.m_in:
                return size

        for i in range(0, self.m_in):
            size += 1

        return size

    def clear(self):
        self.m_in = 0
        self.m_out = 0
        self.m_full = False
        self.m_empty = True

    def is_empty(self):
        return self.m_empty

    def is_full(self):
        return self.m_full

    def to_array(self):
        return self.m_values

    def add(self, o):
        self.enq(o)
        return True

    def contains(self, o):
        for value in self.m_values:
            if value is not None and value == o:
                return True
        return False

    def remove(self, o):
        raise NotImplementedError("remove(Object)")

    def add_all(self, c):
        for element in c:
            self.enq(element)
        return True

    def contains_all(self, c):
        for element in c:
            if not self.contains(element):
                return False
        return True

    def remove_all(self, c):
        raise NotImplementedError("removeAll(Collection)")

    def retain_all(self, c):
        raise NotImplementedError("retainAll(Collection)")

    def __iter__(self):
        return self.QueueIterator(self)

    class QueueIterator:
        def __init__(self, queue):
            self.queue = queue
            self.m_index = queue.m_out

        def __next__(self):
            if self.m_index == self.queue.m_in:
                raise StopIteration
            o = self.queue.m_values[self.m_index]
            self.m_index += 1
            if self.m_index == len(self.queue.m_values):
                self.m_index = 0
            return o

        def __iter__(self):
            return self

    # package level visibility data methods for testing
    def in_(self):
        return self.m_in

    def out(self):
        return self.m_out

    def values(self):
        return self.m_values

class RadixSort:
    timer = Timer(IOFile.VERBOSITY_LEVEL)
    
    @staticmethod
    def radix_sort(v, I, n):
        lines = len(v)
        # # RadixSort.timer.start("create_queues")
        queues = RadixSort.create_queues(n, lines)
        # # RadixSort.timer.end("create_queues")
        pos = len(I) - 1
        
        # # RadixSort.timer.start("loop_radix_sort")
        while pos >= 0:
            # RadixSort.timer.start("loop_radix_sort-internal")
            # for i in range(lines):
            #     q = RadixSort.queue_no(v[i], I[pos])
            #     queues[int(q)].add(v[i])
            i = 0
            while i < lines:
                q = RadixSort.queue_no(v[i], I[pos])
                queues[q].add(v[i])
                i += 1
            # RadixSort.timer.end("loop_radix_sort-internal")
            # # RadixSort.timer.start("loop_radix_sort-restore")
            RadixSort.restore(queues, v)
            # # RadixSort.timer.end("loop_radix_sort-restore")
            pos -= 1
        queues = None
        # call garbage collector
        gc.collect()

    @staticmethod
    def restore(qs, v):
        contv = 0
        for q in qs:
            while not q.is_empty():
                v[contv] = q.deq()
                contv += 1

    @staticmethod
    def create_queues(n, lines):
        queues = []
        i = 0
        while i < int(n):
            queues.append(FIFOQueue(lines))
            i += 1
        return queues

    @staticmethod
    def queue_no(v, pos):
        return int_or_ord_4_digit(v[pos])

class Criteria:
    # probtable = []
    timer = Timer(IOFile.VERBOSITY_LEVEL)

    @staticmethod
    def get_position_of_instances(line, I, A):
        binnumber = ''.join(str(int_or_ord_4_digit(A[line - 1][i])) for i in I)
        return MathRoutines.bin2dec(binnumber)

    @staticmethod
    def equal_instances(line, I, A):
        return all(A[line - 1][i] == A[line][i] for i in I)

    @staticmethod
    def joint_entropy(n, predictors, A, c):
        H = 0
        RadixSort.radix_sort(A, predictors, n)
        lines = len(A)
        pxy = 0
        for j in range(lines):
            if j > 0 and not Criteria.equal_instances(j, predictors, A):
                pxy /= lines
                H -= pxy * (math.log(pxy) / math.log(c))
                pxy = 0
            pxy += 1
        pxy /= lines
        H -= pxy * (math.log(pxy) / math.log(c))
        return H

    @staticmethod
    def instance_criterion(pydx, px, type, alpha, beta, lines, n, dim, c, q):
        H = 0
        if type == "poor_obs":
            if px == 1:
                for k in range(int(c)):
                    pydx[k] = beta if pydx[k] > 0 else (1 - beta) / (c - 1)
            else:
                for k in range(int(c)):
                    pydx[k] /= px
            px /= lines
        elif type == "no_obs":
            for k in range(int(c)):
                pydx[k] /= px
            px += alpha
            px /= (lines + alpha * (n ** dim))
        if q == 0:
            max_prob = max(pydx)
            H = px * (1 - max_prob)
            return H
        elif q == 1:
            H = 0
        else:
            H = 1
        for k in range(int(c)):
            if pydx[k] > 0:
                if q == 1:
                    H -= pydx[k] * (math.log(pydx[k]) / math.log(c))
                else:
                    H -= pydx[k] ** q
        if q != 1:
            H /= (q - 1)
        H *= px
        return H

    @staticmethod
    def MCE_COD(type, alpha, beta, n, c, I, A, q):
        pYdX = [0] * int(c)
        pY = [0] * int(c)
        pX = 0
        H = 0
        HY = 0
        lines = len(A)
        no_obs = n ** len(I)
        # # Criteria.timer.start("Sort")
        RadixSort.radix_sort(A, I, n)
        # # Criteria.timer.end("Sort")
        probtable = [[0] * int(c) for _ in range(int(no_obs))]
        # # Criteria.timer.start("loop_MCE_COD")
        for j in range(lines):
            if j > 0 and not Criteria.equal_instances(j, I, A):
                no_obs -= 1
                position = Criteria.get_position_of_instances(j, I, A)
                probtable[position] = pYdX[:]
                H += Criteria.instance_criterion(pYdX, pX, type, alpha, beta, lines, n, len(I), c, q)
                pYdX = [0] * int(c)
                pX = 0
            pYdX[int_or_ord_4_digit(A[j][-1])] += 1
            pY[int_or_ord_4_digit(A[j][-1])] += 1
            pX += 1
        # # Criteria.timer.end("loop_MCE_COD")
        position = Criteria.get_position_of_instances(lines, I, A)
        probtable[position] = pYdX[:]
        H += Criteria.instance_criterion(pYdX, pX, type, alpha, beta, lines, n, len(I), c, q)
        no_obs -= 1
        HY = Criteria.instance_criterion(pY, lines, "poor_obs", 0, 1, lines, 0, 0, c, q)
        if type == "no_obs" and no_obs > 0:
            penalization = (alpha * no_obs * HY) / (lines + alpha * (n ** len(I)))
            H += penalization
        
        if q >= 0 and q <= 0.00001: # q == 0 -> COD
            return H / HY, probtable
        return H, probtable

class CNMeasurements:
    @staticmethod
    def has_variation(M, is_periodic):
        change = False
        lastcol = len(M[0]) - 1

        for row in range(len(M) - 2):
            if M[row][lastcol] != M[row + 1][lastcol]:
                change = True
                break

        if is_periodic and M[-2][lastcol] != M[-1][lastcol]:
            change = True

        return change

class AGNRoutines:    
    def create_adjacency_matrix(agn):
        if not isinstance(agn, AGN):
            return []
        num_genes = agn.get_nrgenes()
        adjacency_matrix = [[0] * num_genes for _ in range(num_genes)]

        for gt in range(num_genes):
            gene = agn.get_genes()[gt]
            if gene.get_predictorsties() is None or len(gene.get_predictorsties()) == 0:
                for predictor in gene.get_predictors():
                    adjacency_matrix[predictor][gt] = 1
            else:
                for predictorstied in gene.get_predictorsties():
                    for predictor in predictorstied:
                        adjacency_matrix[predictor][gt] = 1

        return adjacency_matrix

    def set_gene_names(agn, genenames):
        if agn.get_nrgenes() == len(genenames):
            for g in range(len(agn.get_genes())):
                name = genenames[g]
                agn.get_genes()[g].set_name(name)
        else:
            IOFile.print_and_log("Error on labeling genes, size does not match.", verbosity=VERBOSE_LEVEL["ERROR"])

    def make_temporal_training_set(agn, target, is_periodic):
        rowsoriginal = len(agn.get_temporalsignalquantized())
        colsoriginal = len(agn.get_temporalsignalquantized()[0])
        rowsts = colsoriginal if is_periodic else colsoriginal - 1

        trainingset = [[''] * rowsoriginal for _ in range(rowsts)]

        for col in range(target):
            for i in range(rowsts):
                newrow = int(agn.get_temporalsignalquantized()[col][i])
                trainingset[i][col] = chr_4_digit(newrow)

        for col in range(target + 1, rowsoriginal):
            for i in range(rowsts):
                newrow = int(agn.get_temporalsignalquantized()[col][i])
                trainingset[i][col - 1] = chr_4_digit(newrow)

        for col in range(1, rowsts + 1):
            i = int(agn.get_temporalsignalquantized()[target][col % colsoriginal])
            trainingset[col - 1][rowsoriginal - 1] = chr_4_digit(i)

        rowsfr = []

        for i in range(len(trainingset)):
            remove = False
            for j in range(len(trainingset[0])):
                if trainingset[i][j] == Preprocessing.SKIP_VALUE:
                    remove = True
                    break
            if remove:
                rowsfr.append(i)

        if rowsfr:
            newtrainingset = [[''] * len(trainingset[0]) for _ in range(len(trainingset) - len(rowsfr))]
            newrow = 0
            for i in range(len(trainingset)):
                if i in rowsfr:
                    newrow += 1
                else:
                    for j in range(len(trainingset[0])):
                        newtrainingset[i - newrow][j] = trainingset[i][j]
            trainingset = newtrainingset

        return trainingset

    def make_steady_state_training_set(agn, target):
        rowsoriginal = len(agn.get_temporalsignalquantized())
        colsoriginal = len(agn.get_temporalsignalquantized()[0])
        rowsts = colsoriginal

        trainingset = [[''] * rowsoriginal for _ in range(rowsts)]

        for col in range(target):
            for i in range(rowsts):
                newrow = int(agn.get_temporalsignalquantized()[col][i])
                trainingset[i][col] = chr_4_digit(newrow)

        for col in range(target + 1, rowsoriginal):
            for i in range(rowsts):
                newrow = int(agn.get_temporalsignalquantized()[col][i])
                trainingset[i][col - 1] = chr_4_digit(newrow)

        for col in range(rowsts):
            i = int(agn.get_temporalsignalquantized()[target][col])
            trainingset[col][rowsoriginal - 1] = chr_4_digit(i)

        rowsfr = []

        for i in range(len(trainingset)):
            remove = False
            for j in range(len(trainingset[0])):
                if trainingset[i][j] == Preprocessing.SKIP_VALUE:
                    remove = True
                    break
            if remove:
                rowsfr.append(i)

        if rowsfr:
            newtrainingset = [[''] * len(trainingset[0]) for _ in range(len(trainingset) - len(rowsfr))]
            newrow = 0
            for i in range(len(trainingset)):
                if i in rowsfr:
                    newrow += 1
                else:
                    for j in range(len(trainingset[0])):
                        newtrainingset[i - newrow][j] = trainingset[i][j]
            trainingset = newtrainingset

        return trainingset
    
    def recover_network_from_temporal_expression(recoveredagn, originalagn, datatype, is_periodic, threshold_entropy, type_entropy, alpha, beta, q_entropy, targets, maxfeatures, searchalgorithm, targetaspredictors, resultsetsize, tiesout, number_of_threads):
        txt = []
        rows = len(recoveredagn.get_temporalsignalquantized())
        IOFile.print_and_log("\n\n")
        txt.append("\n\n")
        timer = Timer(IOFile.VERBOSITY_LEVEL)
        

        if targets is None:
            targets = [str(i) for i in range(rows)]

        # Process a single target and return results
        def process_target(target):
            # Add a short random sleep to better demonstrate concurrency
            local_txt = []
            targetindex = int(target)
            thread_id = threading.get_ident()
            IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} PROCESSING - generating training set", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            
            predictors = []
            ties = []
            sum_of_squares(10**int(os.getenv("COMPLEXITY", "6")))
            # if datatype == 1:
            #     strainingset = AGNRoutines.make_temporal_training_set(recoveredagn, targetindex, is_periodic)
            # else:
            #     strainingset = AGNRoutines.make_steady_state_training_set(recoveredagn, targetindex)
            # fs = FS(strainingset, recoveredagn.get_quantization(), recoveredagn.get_quantization(), type_entropy, alpha, beta, q_entropy, resultsetsize)
            # if not CNMeasurements.has_variation(strainingset, is_periodic):
            #     if targetaspredictors:
            #         message = f"Predictor {targetindex} name {recoveredagn.get_genes()[targetindex].get_name()}, has no variation on its values."
            #         IOFile.print_and_log(message)
            #         local_txt.append(message)
            #     else:
            #         message = f"Target {targetindex} name {recoveredagn.get_genes()[targetindex].get_name()}, has no variation on its values."
            #         IOFile.print_and_log(message)
            #         local_txt.append(message)
            #     return {
            #         "targetindex": targetindex,
            #         "txt": local_txt,
            #         "predictors": [],
            #         "ties": [],
            #         "predictorsties": None,
            #         "probtable": None,
            #         "h_global": 1.0
            #     }
            # else:
            #     # timer.start(f"running_search_algorithm-target_index_{targetindex}")
            #     if searchalgorithm == 1:
            #         IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} PROCESSING - running search algorithm", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            #         fs.run_sfs(False, maxfeatures)
            #     elif searchalgorithm == 3:
            #         IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} PROCESSING - running search algorithm", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            #         fs.run_sffs(maxfeatures, targetindex, recoveredagn)
            #     elif searchalgorithm == 4:
            #         IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} PROCESSING - running search algorithm", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            #         fs.run_sffs_stack(maxfeatures, targetindex, recoveredagn)
            #     elif searchalgorithm == 2:
            #         IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} PROCESSING - running search algorithm", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            #         fs.run_sfs(True, maxfeatures)
            #         s = fs.itmax
            #         fs_prev = FS(strainingset, recoveredagn.get_quantization(), recoveredagn.get_quantization(), type_entropy, alpha, beta, q_entropy, resultsetsize)
            #         for j in range(1, s + 1):
            #             fs = FS(strainingset, recoveredagn.get_quantization(), recoveredagn.get_quantization(), type_entropy, alpha, beta, q_entropy, resultsetsize)
            #             fs.itmax = j
            #             fs.run_exhaustive(0, 0, fs.I)
            #             if not (fs.h_global < fs_prev.h_global):
            #                 fs = fs_prev
            #                 break
            #             fs_prev = fs
            #     # timer.end(f"running_search_algorithm-target_index_{targetindex}")
                
            #     # Prepare text output and collect data to return
            #     if targetaspredictors:
            #         prefix = f"Predictor: {targetindex} name:{recoveredagn.get_genes()[targetindex].get_name()}\nTargets: "
            #         local_txt.append(prefix)
            #         IOFile.print_and_log(prefix, end=" ")
            #     else:
            #         prefix = f"Target: {targetindex} name:{recoveredagn.get_genes()[targetindex].get_name()}\nPredictors: "
            #         local_txt.append(prefix)
            #         IOFile.print_and_log(f"\n{prefix}", end=" ")
                
            #     target_predictors = []
            #     for s in range(len(fs.I)):
            #         predictor_gene = int(fs.I[s])
            #         if predictor_gene >= targetindex:
            #             predictor_gene += 1
            #         if fs.h_global < threshold_entropy:
            #             info = f"{predictor_gene} name:{recoveredagn.get_genes()[predictor_gene].get_name()} "
            #             local_txt.append(info)
            #             IOFile.print_and_log(info, end=" ")
            #             predictors.append(predictor_gene)
            #             target_predictors.append(predictor_gene)
            #         else:
            #             if targetaspredictors:
            #                 info = f"\ntarget {predictor_gene} excluded by threshold. Criterion Function Value = {fs.h_global}"
            #                 local_txt.append(info)
            #                 IOFile.print_and_log(info)
            #             else:
            #                 info = f"\npredictor {predictor_gene} excluded by threshold. Criterion Function Value = {fs.h_global}"
            #                 local_txt.append(info)
            #                 IOFile.print_and_log(info)
                
            #     s = len(fs.I)
            #     predictorsties = None
            #     if (searchalgorithm == 3 or searchalgorithm == 4) and fs.ties[s] and len(fs.ties[s]) > 1 and fs.h_global < threshold_entropy:
            #         local_txt.append("\nPredictors Ties: ")
            #         IOFile.print_and_log("\nPredictors Ties:", end=" ")
            #         predictorsties = [None] * len(fs.ties[s])
            #         for j in range(len(fs.ties[s])):
            #             predictorsties[j] = []
            #             item = fs.ties[s][j]
            #             tie = []
            #             for k in range(len(item)):
            #                 geneindex = int(item[k])
            #                 if geneindex >= targetindex:
            #                     geneindex += 1
            #                 predictorsties[j].append(geneindex)
            #                 local_txt.append(f"{geneindex} name:{recoveredagn.get_genes()[geneindex].get_name()} ")
            #                 IOFile.print_and_log(f"{geneindex} name:{recoveredagn.get_genes()[geneindex].get_name()}", end=" ")
            #                 tie.append(geneindex)
                            
            #             IOFile.print_and_log(" (" + str(fs.jointentropiesties[j]) + ") ", end="\t")
            #             ties.append(tie)
                
            #     if tiesout and originalagn:
            #         originalpredictors = originalagn.get_genes()[targetindex].get_predictors()
            #         IOFile.write_ties(originalagn, tiesout, targetindex, int(originalagn.get_avgedges()), originalagn.get_topology(), originalpredictors, q_entropy, predictors, ties, fs.h_global, False)
                
            #     IOFile.print_and_log(f"\n\nCriterion Function Value: {fs.h_global}")
            #     local_txt.append(f"\nCriterion Function Value: {fs.h_global}\n")
                
            #     return {
            #         "targetindex": targetindex,
            #         "txt": local_txt,
            #         "predictors": target_predictors,
            #         "ties": ties,
            #         "predictorsties": predictorsties,
            #         "probtable": fs.probtable,
            #         "h_global": fs.h_global
            #     }

        results = [None] * len(targets)  # Preallocate the results list with None values

        def process_target_wrapper(target, index):
            thread_id = threading.get_ident()
            start_time = time.time()
            IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} STARTED at {start_time}", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            result = process_target(target)
            end_time = time.time()
            IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} ENDED at {end_time} (Duration: {end_time - start_time:.4f}s)", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            results[index] = result
            
        def process_group(group, offet_index):
            for i, target in enumerate(group):
                index = offet_index + i
                process_target_wrapper(target, index)

        threads = []
        
        # for i in range(0, len(targets), group_size):
        #     group = targets[i:i + group_size]

        num_processes_per_thread = len(targets) // number_of_threads + (1 if len(targets) % number_of_threads > 0 else 0)
        
        for i in range(0, number_of_threads * num_processes_per_thread, num_processes_per_thread):
            group = targets[i:i + num_processes_per_thread]
            
            IOFile.print_and_log(f"Starting group {i//num_processes_per_thread + 1} with targets: {group}", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
            
            # for j, target in enumerate(group):
            #     index = i + j
            #     thread = threading.Thread(target=process_target_wrapper, args=(target, index, recoveredagn), name=f"Target-{target}-main_from_cli")
            #     threads.append(thread)
            #     thread.start()
            thread = threading.Thread(target=process_group, args=(group, i), name=f"Group-{i//num_processes_per_thread + 1}-main_from_cli")
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
        
        IOFile.print_and_log(f"Completed group {i//num_processes_per_thread + 1}", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])

        # for result in results:
        #     targetindex = result["targetindex"]
        #     txt.extend(result["txt"])
            
        #     # Update the AGN object with results for this target
        #     if result["predictors"]:
        #         for predictor in result["predictors"]:
        #             recoveredagn.get_genes()[targetindex].add_predictor(predictor, result["h_global"])
        #             recoveredagn.get_genes()[predictor].add_target(targetindex)
                
        #         if result["predictorsties"]:
        #             recoveredagn.get_genes()[targetindex].set_predictorsties(result["predictorsties"])
                
        #         if result["probtable"]:
        #             recoveredagn.get_genes()[targetindex].set_probtable(result["probtable"])

        IOFile.print_and_log(f"Completed all targets", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
        return "\n".join(txt)

class Classifier:
    def __init__(self):
        self.table = []
        self.instances = []
        self.labels = []

    def equal_instances(self, line, I, A):
        for i in I:
            if A[line - 1][i] != A[line][i]:
                return False
        return True

    def index_max_value(self, v):
        index_max = -1
        maximum = float('-inf')
        for i, value in enumerate(v):
            if value > maximum:
                index_max = i
                maximum = value
        ties = [i for i, value in enumerate(v) if value == maximum]
        if len(ties) > 1:
            return random.choice(ties)
        return index_max

    def instance_index(self, sample, I, n):
        instance = 0
        dim = len(I)
        for i in range(dim):
            instance += int_or_ord_4_digit(sample[I[dim - i - 1]]) * (n ** i)
        return instance

    def add_table_line(self, sample, I, pYdX, pX, n, c):
        instance = self.instance_index(sample, I, n)
        table_line = [pYdX[k] for k in range(int(c))]
        for k in range(int(c)):
            pYdX[k] = 0
        pX = 0
        self.instances.append(instance)
        self.table.append(table_line)

    def binary_search(self, value):
        start, end = 0, len(self.instances) - 1
        while start <= end:
            v = (start + end) // 2
            if self.instances[v] == value:
                return v
            elif self.instances[v] < value:
                start = v + 1
            else:
                end = v - 1
        return -1

    def instance_vector(self, instance_index, n, d):
        V = [0] * d
        for i in range(d - 1, -1, -1):
            if instance_index == 0:
                break
            V[i] = int(instance_index % n)
            instance_index = math.floor(instance_index / n)
        return V

    def euclidean_distance(self, v1, v2):
        return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))

    def nearest_neighbors(self, instance_index, n, d, c):
        instance_values = self.instance_vector(instance_index, n, d)
        distances = [self.euclidean_distance(instance_values, self.instance_vector(inst, n, d)) for inst in self.instances]
        pYdX = [0] * int(c)
        while True:
            min_dist = min(distances)
            for i, dist in enumerate(distances):
                if dist == min_dist:
                    for j in range(int(c)):
                        pYdX[j] += self.table[i][j]
                    distances[i] = float('inf')
            index_max = self.index_max_value(pYdX)
            if index_max > -1:
                return index_max

    def classifier_table(self, A, I, n, c):
        lines = len(A)
        pX = 0
        pYdX = [0] * int(c)
        RadixSort.radix_sort(A, I, n)
        for j in range(lines):
            if j > 0 and not self.equal_instances(j, I, A):
                self.add_table_line(A[j - 1], I, pYdX, pX, n, c)
            pYdX[int_or_ord_4_digit(A[j][-1])] += 1
            pX += 1
        self.add_table_line(A[lines - 1], I, pYdX, pX, n, c)

    def classify_test_samples(self, A, I, n, c):
        lines = len(A)
        self.labels = [0] * lines
        test_instances = [self.instance_index(A[i], I, n) for i in range(lines)]
        for i in range(lines):
            index = self.binary_search(test_instances[i])
            if index == -1:
                self.labels[i] = self.nearest_neighbors(test_instances[i], n, len(I), c)
            else:
                self.labels[i] = self.index_max_value(self.table[index])
                if self.labels[i] == -1:
                    self.labels[i] = self.nearest_neighbors(test_instances[i], n, len(I), c)
        return test_instances

def main():
    load_dotenv(override=True)
    timer = Timer()

    # Configuration parameters
    output_folder = os.getenv("OUTPUT_FOLDER")
    verbosity_level = int(os.getenv("VERBOSITY_LEVEL", ))
    IOFile.set_verbosity(verbosity_level)
    # timer.set_verbosity(verbosity_level)
    
    if output_folder is not None and output_folder != "":
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    input_file_path = os.getenv("INPUT_FILE_PATH")
    has_column_description = os.getenv("ARE_COLUMNS_DESCRIPTIVE") == "true"
    has_data_titles_columns = os.getenv("ARE_TITLES_ON_FIRST_COLUMN") == "true"
    has_transpose_matrix = os.getenv("TRANSPOSE_MATRIX") == "true"

    quantization_value = int(os.getenv("QUANTIZATION_VALUE"))
    has_labels = 1 if os.getenv("HAS_LABELS_CLASSES_ON_LAST_COLUMN") == "true" else 0
    is_to_look_for_cycles = os.getenv("LOOK_FOR_CYCLES") == "true"
    quantization_type = int(os.getenv("APPLY_QUANTIZATION_TYPE"))
    is_to_save_quantized_data = os.getenv("SAVE_QUANTIZED_DATA") == "true"

    is_to_execute_feature_selection = os.getenv("EXECUTE_FEATURE_SELECTION") == "true"
    criteria_function_feature_selection = int(os.getenv("CRITERION_FUNCTION_FEATURE_SELECTION", "0"))
    q_entropy_feature_selection = float(os.getenv("Q_ENTROPY_FEATURE_SELECTION", "1.0"))
    penalization_method_feature_selection = int(os.getenv("PENALIZATION_METHOD_FEATURE_SELECTION", "0"))
    alpha_feature_selection = float(os.getenv("ALPHA_FEATURE_SELECTION", "1.0"))
    beta_feature_selection = float(os.getenv("BETA_FEATURE_SELECTION", "80"))
    input_test_set_feature_selection = os.getenv("INPUT_TEST_SET_FEATURE_SELECTION", None)
    search_method_feature_selection = int(os.getenv("SEARCH_METHOD_FEATURE_SELECTION", "1"))
    maximum_set_size_feature_selection = int(os.getenv("MAXIMUM_SET_SIZE_FEATURE_SELECTION", "3"))
    maximum_result_list_size_feature_selection = int(os.getenv("SIZE_OF_THE_RESULT_LIST_FEATURE_SELECTION", "3"))

    number_of_threads = int(os.getenv("NUMBER_OF_THREADS", "1"))
    target_indexes = os.getenv("TARGET_INDEXES", None)
    is_targets_as_predictors = os.getenv("TARGETS_AS_PREDICTORS") == "true"
    is_time_series_data = os.getenv("TIME_SERIES_DATA") == "true"
    is_it_periodic = os.getenv("IS_IT_PERIODIC") == "true"
    threshold = float(os.getenv("THRESHOLD", "0.3"))
    is_to_save_final_data = os.getenv("SAVE_FINAL_DATA") == "true"

    # Config Outputs
    prefix = str(datetime.now().timestamp()).replace(".", "_")
    if output_folder is None or output_folder == "":
        is_to_save_quantized_data = False
        is_to_save_final_data = False
    
    # delimiter = " \t\n\r\f;"
    delimiter = "\t"
    Mo = None
    Md = None
    lines = 0
    columns = 0
    datatitles = None
    featurestitles = None
    flag_quantization = False
    recoverednetwork = None

    def read_data_action_performed():
        nonlocal Mo, Md, lines, columns, datatitles, featurestitles, flag_quantization
        datatitles = None
        featurestitles = None
        flag_quantization = False
        
        start_row = 0
        start_column = 0
        try:
            if input_file_path.endswith('agn'):
                IOFile.print_and_log("Skipping reading AGN file")
            else:
                if has_column_description:
                    featurestitles = IOFile.read_data_first_row(input_file_path, 0, 0, delimiter)
                    start_row = 1
                if has_data_titles_columns:
                    datatitles = IOFile.read_data_first_column(input_file_path, start_row, delimiter)
                    start_column = 1
                Mo = IOFile.read_matrix(input_file_path, start_row, start_column, delimiter)
            if has_transpose_matrix:
                Mo = list(map(list, zip(*Mo)))
        except:
          IOFile.print_and_log('Something went wrong')
        finally:
            lines = len(Mo)
            columns = len(Mo[0])
            Md = Preprocessing.copy_matrix(Mo)

    def apply_quantization_action(qtvalues, type):
        nonlocal Md, flag_quantization
        if Mo is not None:
            Md = Preprocessing.copy_matrix(Mo)
            if type == 1:
                Preprocessing.quantize_columns(Md, qtvalues, True, has_labels)
            else:
                Preprocessing.quantize_rows(Md, qtvalues, True, has_labels)
            flag_quantization = True

            if is_to_save_quantized_data:
                IOFile.write_matrix(f"{output_folder}/quantized_data/{prefix}-quantized_data.txt", Md, delimiter)
        else:
            IOFile.print_and_log("Execution Error: Select and read input file first.", verbosity=VERBOSE_LEVEL["ERROR"])

    def execute_feature_selection_action_performed():
        alpha = alpha_feature_selection
        q_entropy = q_entropy_feature_selection
        if q_entropy < 0 or alpha < 0:
            IOFile.print_and_log("Error on parameter value: The values of q-entropy and Alpha must be positives.", verbosity=VERBOSE_LEVEL["ERROR"])
            return
        if search_method_feature_selection > 0 and search_method_feature_selection <= 3:
            thread = threading.Thread(target=execute_feature_selection, args=(search_method_feature_selection,))
            thread.name = "SE"
            thread.start()
            
            thread.join()
        else:
            IOFile.print_and_log("Error on parameter value: The search method must be selected.", verbosity=VERBOSE_LEVEL["ERROR"])

    def execute_feature_selection(selector):
        # timer.start("feature_selection_inside_thread")
        penalization_type = "no_obs" if penalization_method_feature_selection == 0 else "poor_obs"
        alpha = float(alpha_feature_selection)
        q_entropy = float(q_entropy_feature_selection)
        beta = beta_feature_selection / 100
        if criteria_function_feature_selection == 1:
            q_entropy = 0
        if q_entropy < 0 or alpha < 0:
            IOFile.print_and_log("Error on parameter value: The values of q-entropy and Alpha must be positives.", verbosity=VERBOSE_LEVEL["ERROR"])
            return
        n = max(max(row[:-1]) for row in Md) + 1
        c = max(row[-1] for row in Md) + 1
        
        # strainingset = [[str(value) for value in row] for row in Md]
        strainingset = MathRoutines.float2char(Md)
        if input_test_set_feature_selection:
            stestset = IOFile.read_matrix(input_test_set_feature_selection, 0, 0, delimiter)
        else:
            # stestset = [[str(value) for value in row] for row in Md]
            stestset = MathRoutines.float2char(Md)
        
        resultsetsize = maximum_result_list_size_feature_selection
        if resultsetsize < 1:
            IOFile.print_and_log("Error on parameter value: The Size of the Result List must be a integer value greater or equal to 1.", verbosity=VERBOSE_LEVEL["ERROR"])
            return
        fs = FS(strainingset, n, c, penalization_type, alpha, beta, q_entropy, resultsetsize)
        maxfeatures = maximum_set_size_feature_selection
        if maxfeatures <= 0:
            IOFile.print_and_log("Error on parameter value: The Maximum Set Size be a integer value greater or equal to 1.", verbosity=VERBOSE_LEVEL["ERROR"])
            return
        if selector == 1:
            fs.run_sfs(False, maxfeatures)
        elif selector == 3:
            fs.run_sffs(maxfeatures, -1, None)
        elif selector == 2:
            fs.run_sfs(True, maxfeatures)
            itmax = fs.itmax
            if itmax < maxfeatures:
                itmax = maxfeatures
            combinations = sum(math.comb(columns - 1, i) for i in range(1, itmax + 1))
            estimated_time = (0.0062 + 3.2334e-7 * len(strainingset)) * combinations * math.log2(combinations)
            IOFile.print_and_log(f"Estimated time to finish: {estimated_time} s")
            fs_prev = FS(strainingset, n, c, penalization_type, alpha, beta, q_entropy, resultsetsize)
            for i in range(1, itmax + 1):
                IOFile.print_and_log(f"Iteration {i}")
                fs = FS(strainingset, n, c, penalization_type, alpha, beta, q_entropy, resultsetsize)
                fs.itmax = i
                fs.run_exhaustive(0, 0, fs.I)
                if fs.h_global < fs_prev.h_global:
                    fs_prev = fs
                else:
                    fs = fs_prev
                    break
        for i, result in enumerate(fs.resultlist):
            fsvalue = result[0]
            IOFile.print_and_log(f"{i + 1}st Global Criterion Function Value: {fsvalue}")
            IOFile.print_and_log("Selected Features: ", result[1])
        clas = Classifier()
        clas.classifier_table(strainingset, fs.I, n, c)
        for i, table_line in enumerate(clas.table):
            instance = clas.instances[i]
            IOFile.print_and_log(instance, table_line)
        instances = clas.classify_test_samples(stestset, fs.I, n, c)
        
        IOFile.print_and_log("Correct Labels  -  Classified Labels - Classification Instances\n(Considering the first selected features)")
        hits = 0
        for i in range(len(clas.labels)):
            correct_label = int_or_ord_4_digit(stestset[i][-1])
            classified_label = clas.labels[i]
            if correct_label == classified_label:
                hits += 1
            IOFile.print_and_log(f"{correct_label}  -  {classified_label}  -  {instances[i]}")
        # hits = sum(1 for i in range(len(clas.labels)) if int(stestset[i][-1]) == clas.labels[i])
        hit_rate = hits / len(clas.labels)
            
        IOFile.print_and_log(f"rate of hits = {hit_rate}")
        # timer.end("feature_selection_inside_thread")

    def network_inference_action_performed():
        nonlocal Md
        threshold_entropy = float(threshold)
        type_entropy = "no_obs" if penalization_method_feature_selection == 0 else "poor_obs"
        alpha = float(alpha_feature_selection)
        q_entropy = float(q_entropy_feature_selection)
        beta = beta_feature_selection / 100
        search_alg = search_method_feature_selection
        targets = target_indexes.split() if target_indexes else None
        if is_targets_as_predictors:
            Md = Preprocessing.invert_columns(Md)
        resultsetsize = 1
        n = int(max(max(row) for row in Md)) + 1
        recoverednetwork = AGN(len(Md), len(Md[0]), n)
        recoverednetwork.set_temporalsignal(Mo)
        recoverednetwork.set_temporalsignalquantized(Md)
        if featurestitles:
            recoverednetwork.set_labelstemporalsignal(featurestitles)
        if datatitles:
            AGNRoutines.set_gene_names(recoverednetwork, datatitles)
        datatype = 1 if is_time_series_data else 2
        
        # Create timing directory if it doesn't exist
        if not os.path.exists("timing"):
            os.makedirs("timing")
            
        # timer.start("network_inference")
        txt = AGNRoutines.recover_network_from_temporal_expression(
            recoverednetwork,
            None,
            datatype,
            is_it_periodic,
            threshold_entropy,
            type_entropy,
            alpha,
            beta,
            q_entropy,
            targets,
            maximum_set_size_feature_selection,
            search_alg,
            is_targets_as_predictors,
            resultsetsize,
            None,
            number_of_threads
        )
        if is_to_save_final_data:
            IOFile.write_matrix(f"{output_folder}/final_data/{prefix}-final_data.txt", AGNRoutines.create_adjacency_matrix(recoverednetwork), delimiter)
        
        IOFile.print_and_log(txt)

    # timer.start("read_data")
    read_data_action_performed()
    # timer.end("read_data")

    if quantization_type > 0 and quantization_type <= 2:
        # timer.start("apply_quantization")
        apply_quantization_action(quantization_value, quantization_type)
        # timer.end("apply_quantization")

    if is_to_look_for_cycles and Md is not None:
        # timer.start("find_cycle")
        CNMeasurements.find_cycle(Md)
        # timer.end("find_cycle")

    if is_to_execute_feature_selection:
        # timer.start("execute_feature_selection")
        execute_feature_selection_action_performed()
        # timer.end("execute_feature_selection")

    # timer.start("network_inference")
    IOFile.print_and_log(f"{datetime.now()}; start network inference - start", path="timing/timers.log", verbosity=VERBOSE_LEVEL["NONE"])
    network_inference_action_performed()
    IOFile.print_and_log(f"{datetime.now()}; start network inference - end", path="timing/timers.log", verbosity=VERBOSE_LEVEL["NONE"])
    # timer.end("network_inference")

if __name__ == "__main__":
    main()