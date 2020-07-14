import csv

with open('question-4-train-labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    negative_count = 0
    positive_count = 0
    neutral_count = 0
    negative_list = []
    potisive_list = []
    neutral_list = []

    with open('question-4-train-features.csv') as csv_hop:
        hop = csv.reader(csv_hop, delimiter=' ')

        for row in csv_reader:
            for vector in hop:
                if row[0] == "negative":
                    negative_list.append(vector[0])
                    negative_count += 1
                    break

                elif row[0] == "positive":
                    potisive_list.append(vector[0])
                    positive_count += 1
                    break
                else:
                    neutral_list.append(vector[0])
                    neutral_count += 1
                    break

print(f'Negative Count is {negative_count}.')
print(f'Positive Count is {positive_count}.')
print(f'Neuteal Count is {neutral_count}.')


def mutinomial(array_list, count):
    j = 0
    p = 11443
    n = count
    hop = []
    while (j < p):
        i = 0
        sum = 0
        while (i < n):
            sum = sum + int(array_list[i][j])
            i = i + 1
        hop.append(sum)
        j = j + 2

    print(hop)
    return hop


def bernoulli(array_list, count):
    j = 0
    p = 11443
    n = count
    hop = []
    while j < p:
        i = 0
        sum = 0
        while i < n:
            if int(array_list[i][j]) > 0:
                sum = sum + 1
            i = i + 1
        hop.append(sum)
        j = j + 2

    print(hop)
    return hop

negative_multinomial = mutinomial(negative_list, negative_count)
positive_multinomial = mutinomial(potisive_list, positive_count)
neutral_multinomial = mutinomial(neutral_list, neutral_count)

negative_bernoulli = bernoulli(negative_list, negative_count)
positive_bernoulli = bernoulli(potisive_list, positive_count)
neutral_bernoulli = bernoulli(neutral_list, neutral_count)

def find_mean(array_list):
    myInt = 11444
    newList = [x / myInt for x in array_list]
    return newList

negativeM_mean = find_mean(negative_multinomial)
positiveM_mean = find_mean(positive_multinomial)
neutralM_mean = find_mean(neutral_multinomial)
neutralM_mean[0]

negativeB_mean = find_mean(negative_bernoulli)
positiveB_mean = find_mean(positive_bernoulli)
neutralB_mean = find_mean(neutral_bernoulli)
neutralB_mean[0]

def classify(meanArray, sample, size):
    yMean = size / 11712
    classification = 1
    i = 0
    j = 0
    hop = 1
    while(i < 5722):
        classification = classification * (float(sample[j]) * float(meanArray[i]) + (1 - float(sample[j])) * (1 - float(meanArray[i])))
        #classification = classification * (float(sample[j]) * float(meanArray[i]))
        i = i + 1
        j = j + 2
    hop = classification * yMean
    return hop


sample = []
with open('question-4-test-features.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    i = 0
    while (i < 2928):
        for row in csv_reader:
            sample.append(row[0])
        i = i + 1

labels = []
with open('question-4-test-labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    i = 0
    while (i < 2928):
        for row in csv_reader:
            labels.append(row[0])
        i = i + 1

accurancyM = 0
i = 0
while (i < 2928):
    negative = classify(negativeM_mean, sample[i], negative_count)
    positive = classify(positiveM_mean, sample[i], positive_count)
    neutral = classify(neutralM_mean, sample[i], neutral_count)
    if (labels[i] == "negative" and (negative > neutral and negative > positive)):
        accurancyM = accurancyM + 1

    elif (labels[i] == "positive" and (positive > neutral and positive > negative)):
        accurancyM = accurancyM + 1

    elif (labels[i] == "neutral" and (neutral > negative and neutral > positive)):
        accurancyM = accurancyM + 1
    i = i + 1

accurancyB = 0
i = 0
while (i < 2928):
    negative = classify(negativeB_mean, sample[i], negative_count)
    positive = classify(positiveB_mean, sample[i], positive_count)
    neutral = classify(neutralB_mean, sample[i], neutral_count)
    if (labels[i] == "negative" and (negative > neutral and negative > positive)):
        accurancyB = accurancyB + 1

    elif (labels[i] == "positive" and (positive > neutral and positive > negative)):
        accurancyB = accurancyB + 1

    elif (labels[i] == "neutral" and (neutral > negative and neutral > positive)):
        accurancyB = accurancyB + 1
    i = i + 1

print("Multi Accurancy is ")
print(accurancyM / 2928)
print("\n")
print("Bernoulli Accurancy is ")
print(accurancyB / 2928)