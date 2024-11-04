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
        np.savetxt(filepath, matrix, delimiter=delimiter)

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

class AGN:
    def __init__(self, nrgenes, signalsize, base):
        self.nrgenes = nrgenes
        self.signalsize = signalsize
        self.base = base
        # Initialize other attributes

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def set_lowthreshold(self, lowthreshold):
        self.lowthreshold = lowthreshold

    def set_hithreshold(self, hithreshold):
        self.hithreshold = hithreshold

    def set_temporalsignal(self, expressiondata):
        self.temporalsignal = expressiondata

    def set_temporalsignalquantized(self, quantizeddata):
        self.temporalsignalquantized = quantizeddata

    def set_temporalsignalnormalized(self, normalizeddata):
        self.temporalsignalnormalized = normalizeddata

    def set_labelstemporalsignal(self, featurestitles):
        self.labelstemporalsignal = featurestitles

class AGNRoutines:
    @staticmethod
    def set_gene_names(agn, genenames):
        agn.genenames = genenames

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

    # recoverednetwork = AGN(nrgenes, signalsize, 2)
    # recoverednetwork.set_mean(mean)
    # recoverednetwork.set_std(std)
    # recoverednetwork.set_lowthreshold(lowthreshold)
    # recoverednetwork.set_hithreshold(hithreshold)
    # recoverednetwork.set_temporalsignal(expressiondata)
    # recoverednetwork.set_temporalsignalquantized(quantizeddata)
    # recoverednetwork.set_temporalsignalnormalized(normalizeddata)
    # recoverednetwork.set_labelstemporalsignal(featurestitles)
    # AGNRoutines.set_gene_names(recoverednetwork, genenames)

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