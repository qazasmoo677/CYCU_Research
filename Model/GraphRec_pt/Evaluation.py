import numpy as np

class Evaluation():
    def checkNeedCalcUser(self, predicted, actual):
        if len(actual) > len(predicted):
            return True
        return False

    def calcPrecisionAndRecall(self, predicted, actual):
        if len(actual) > len(predicted):
            predict_set = set(predicted)
            actual_set = set(actual[:len(predicted)])
            
            true_positives = len(predict_set.intersection(actual_set))
            precision = true_positives / len(predict_set) if predict_set else 0.0
            recall = true_positives / len(actual_set) if actual_set else 0.0

            return precision, recall
        else:
            return 0.0, 0.0

    def calcMAP(self, predicted, actual):
        if len(actual) > len(predicted):
            total_ap = 0.0
            for i, p in enumerate(predicted):
                if p in actual:
                    relevant_count = len([x for x in predicted[:i + 1] if x in actual])
                    precision_at_i = relevant_count / (i + 1)
                    total_ap += precision_at_i

            map_score = total_ap / len(actual) if len(actual) > 0 else 0.0
            return map_score
        else:
            return 0.0

    def calcNDCG(self, predicted, actual):
        if len(actual) > len(predicted):
            actual_dict = {item: i+1 for i, item in enumerate(actual)}

            relevance = []
            for i, item in enumerate(predicted):
                if item in actual_dict:
                    relevance.append(1.0 / actual_dict[item])
                else:
                    relevance.append(0.0)

            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            return ndcg
        else:
            return 0.0