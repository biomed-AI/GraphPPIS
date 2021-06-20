import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from GraphPPIS_model import *

# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/slow_version/"


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, graphs = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda())
                graphs = Variable(graphs.cuda())
                y_true = Variable(labels.cuda())
            else:
                node_features = Variable(node_features)
                graphs = Variable(graphs)
                y_true = Variable(labels)

            node_features = torch.squeeze(node_features)
            graphs = torch.squeeze(graphs)
            y_true = torch.squeeze(y_true)

            y_pred = model(node_features, graphs)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))

        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)

        result_test = analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])
        print()

        # Export prediction
        # with open(model_name.split(".")[0] + "_pred.pkl", "wb") as f:
            # pickle.dump(pred_dict, f)


def test_one_dataset(dataset):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe)


def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_331.pkl", "rb") as f:
        Test_331 = pickle.load(f)

    with open(Dataset_Path + "UBtest_31.pkl", "rb") as f:
        UBtest_31 = pickle.load(f)

    Btest_31 = {}
    with open(Dataset_Path + "bound_unbound_mapping.txt", "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        bound_ID, unbound_ID, _ = line.strip().split()
        Btest_31[bound_ID] = Test_60[bound_ID]

    print("Evaluate GraphPPIS on Test_60")
    test_one_dataset(Test_60)

    print("Evaluate GraphPPIS on Test_331")
    test_one_dataset(Test_331)

    print("Evaluate GraphPPIS on Btest_31")
    test_one_dataset(Btest_31)

    print("Evaluate GraphPPIS on UBtest_31")
    test_one_dataset(UBtest_31)


if __name__ == "__main__":
    main()
