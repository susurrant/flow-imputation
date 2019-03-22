
import numpy as np


class AccuracySummary:

    def __init__(self, p, r):
        self.results = {'MAE': 0,
                        'MAPE': 0,
                        'MSE': 0,
                        'RMSE': 0,
                        'CPC:': 0,
                        'SSI': 0,
                        'R^2': 0}
        self.p = p
        self.r = r
        p = np.array(p)
        r = np.array(r)

        self.results['MAE'] = np.mean(np.abs(r-p))

        c1 = 0
        mape = 0
        c2 = 0
        ssi = 0
        for i in range(p.shape[0]):
            if r[i]:
                mape += np.abs((r[i]-p[i])/r[i])
                c1 += 1
            if r[i]+p[i]:
                ssi += min(r[i], p[i])/(r[i]+p[i])
                c2 += 1
        self.results['MAPE'] = mape*100/c1
        self.results['SSI'] = ssi*2/(c2^2)

        self.results['R^2'] = 1 - np.sum(np.square(r - p))/np.sum(np.square(r-np.mean(r)))

        self.results['MSE'] = np.mean(np.square(r-p))
        self.results['RMSE'] = np.sqrt(self.results['MSE'])

        stack = np.column_stack((p, r))
        self.results['CPC'] = 2 * np.sum(np.min(stack, axis=1)) / np.sum(stack)

    def accuracy_string(self):
        return 'MSE'

    def pretty_print(self):
        print('real:', self.r[:20])
        print('pred:', list(map(int, self.p[:20])))
        for item in self.results.items():
            print(item[0], end='\t')
            print(str(round(item[1],3)), end='\n')


class AccuracyScore:

    def __init__(self, pred, real):
        self.pred = pred
        self.real = real

    def summarize(self):
        summary = self.get_summary()
        summary.pretty_print()

    def get_summary(self):
        return AccuracySummary(self.pred, self.real)

        
class Scorer:

    def __init__(self, settings, model):
        self.settings = settings
        self.model = model

    def register_model(self, model):  #model_builder.build_decoder: BilinearDiag object inheriting from Model
        self.model = model

    def compute_accuracy_scores(self, triples, verbose=False):
        pred = self.model.score(triples[:,:3])
        real = triples[:,3]
        score = AccuracyScore(pred, real)

        if verbose:
            print("Evaluating accuracy for test triplets...")

        return score

    def compute_scores(self, triples, verbose=False):
        if self.settings['Metric'] == 'Accuracy':
            return self.compute_accuracy_scores(triples, verbose=verbose)

    def dump_all_scores(self, triples, filename):
        pass



