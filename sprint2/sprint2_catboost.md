very basic pipeline with : 
    (**numeric_cols=numeric_cols,
    log_cols=log_cols,
    cat_cols=cat_cols_for_cb,
    auto_class_weights="Balanced",
    iterations=600,
    depth=8,
    learning_rate=0.08,
    random_state=42**)
Hold-out metrics:
accuracy: 0.7034
balanced_accuracy: 0.6989
f1_macro: 0.6256
f1_weighted: 0.7304
precision_macro: 0.6294
precision_weighted: 0.7817
recall_macro: 0.6989
recall_weighted: 0.7034
log_loss: 0.6593

Classification report (hold-out):
                          precision    recall  f1-score   support

             functional     0.8246    0.6971    0.7555      6452
functional needs repair     0.2402    0.6837    0.3555       863
         non functional     0.8234    0.7159    0.7659      4565

               accuracy                         0.7034     11880
              macro avg     0.6294    0.6989    0.6256     11880
           weighted avg     0.7817    0.7034    0.7304     11880


Confusion matrix labels: [np.str_('functional'), np.str_('functional needs repair'), np.str_('non functional')]
[[4498 1341  613]
 [ 185  590   88]
 [ 772  525 3268]]