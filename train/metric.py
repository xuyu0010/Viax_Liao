"""
Metric function library
Author: Yunpeng Chen
* most of the code are inspired by MXNet
"""
import logging
import numpy as np

class EvalMetric(object):       #评价标准

    def __init__(self, name, **kwargs):
        self.name = str(name)
        self.reset()

    def update(self, preds, labels, losses):        #更新metric
        raise NotImplementedError()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):      ## get方法是计算当前的evaluation result。比如说你设定每计算20个batch后就计算一次准确率等结果，那么这个结果的计算就是调用这个get方法
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):       #Returns zipped name and value pairs.
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    #用来判断labels和preds的shape是否一致，因为labels和preds都是list，而且一般这个list中只包含一个NDArray，比如说你的batch size是16，类别数是1000，那么labels中的NDArray就是16*1，preds中的NDArray就是16*1000。
    def check_label_shapes(self, preds, labels):
        # raise if the shape is inconsistent
        if (type(labels) is list) and (type(preds) is list):
            label_shape, pred_shape = len(labels), len(preds)
        else:
            label_shape, pred_shape = labels.shape[0], preds.shape[0]

        if label_shape != pred_shape:
            raise NotImplementedError("")


class MetricList(EvalMetric):       
    """Handle multiple evaluation metric
    """
    def __init__(self, *args, name="metric_list"):
        assert all([issubclass(type(x), EvalMetric) for x in args]), \
            "MetricList input is illegal: {}".format(args)
        self.metrics = [metric for metric in args]
        super(MetricList, self).__init__(name=name)

    def update(self, preds, labels, losses=None):       #labels和preds都是一个NDArray的列表
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels
        losses = [losses] if type(losses) is not list else losses

        for metric in self.metrics:
            metric.update(preds, labels, losses)

    def reset(self):
        if hasattr(self, 'metrics'):
            for metric in self.metrics:
                metric.reset()
        else:
            logging.warning("No metric defined.")

    def get(self):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get())
        return ouputs

    def get_name_value(self):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get_name_value())        
        return ouputs


####################
# COMMON METRICS
####################

class Accuracy(EvalMetric):
    """Computes accuracy classification score.
    """
    def __init__(self, name='accuracy', topk=1):
        super(Accuracy, self).__init__(name)    ## super这个函数是调用基类mx.metric.EvalMetric的__init__函数，__init__函数括号中的变量是要传递给基类的__init__函数的变量。super()括号中的Accuracy表示类名称。
        self.topk = topk
#topk函数，可以将高维数组沿某一维度（该维度共N项），选出最大（最小）的K项并排序。返回排序结果和index信息。
    def update(self, preds, labels, losses):        #Updates the internal evaluation result.
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels

        self.check_label_shapes(preds, labels)
        for pred, label in zip(preds, labels):  # # zip函数可以将输入的两个list的对应位置的值变成一个元组(tuple)，这样每个tuple就包含两个值，这两个值在这里都是NDArray格式。又因为pred_label的shape和label的shape是不一样的，所以都会进入下面这个if语句，也就是先将pred_label按行求出最大值所在的index，然后pred_label就和label是相同shape的NDArray了。
            assert self.topk <= pred.shape[1], \
                "topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
            _, pred_topk = pred.topk(self.topk, 1, True, True)

            pred_topk = pred_topk.t()
            correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))

            self.sum_metric += float(correct.view(-1).float().sum(0, keepdim=True).numpy())
            self.num_inst += label.shape[0]


class Loss(EvalMetric):
    """Dummy metric for directly printing loss.
    """        
    def __init__(self, name='loss'):
        super(Loss, self).__init__(name)

    def update(self, preds, labels, losses):
        assert losses is not None, "Loss undefined."
        for loss in losses:
            self.sum_metric += float(loss.numpy().sum())
            # self.num_inst += loss.shape[0]
            self.num_inst += loss.numpy().size # for pytorch040


if __name__ == "__main__":
    import torch

    # Test Accuracy
    predicts = [torch.from_numpy(np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]]))]
    labels   = [torch.from_numpy(np.array([   0,            1,          1 ]))]
    losses   = [torch.from_numpy(np.array([   0.3,       0.4,       0.5   ]))]

    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("input pred:  {}".format(predicts))
    logging.debug("input label: {}".format(labels))
    logging.debug("input loss: {}".format(labels))

    acc = Accuracy()

    acc.update(preds=predicts, labels=labels, losses=losses)

    logging.info(acc.get())

    # Test MetricList
    metrics = MetricList(Loss(name="ce-loss"),
                         Accuracy(topk=1, name="acc-top1"), 
                         Accuracy(topk=2, name="acc-top2"), 
                         )
    metrics.update(preds=predicts, labels=labels, losses=losses)

    logging.info("------------")
    logging.info(metrics.get())
    acc.get_name_value()
