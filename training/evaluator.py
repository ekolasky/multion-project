import json

class Evaluator():
    '''
    Evaluator gets the error metrics for a list of example evaluations. It does this by finding matching entities between the 
    predictions and labels by name. The user can use the Evaluator to visualize matching and non-matching entities and, if necessary,
    link them to get new error metrics.
    '''
    def __init__(self, example_evals):
        '''
        Initializes the evaluator with a list of example evaluations. These example evaluations are computed using the evaluate_model
        script, and saved to a JSON file in the evals folder.
        '''
        
        example_eval_objs = []
        # Run through each example
        for example_eval in example_evals:
            example_eval_obj = ExampleEvalObject(example_eval)
            example_eval_objs.append(example_eval_obj)

        error_metrics = self.get_error_metrics(example_eval_objs)
        self.example_evals = example_eval_objs
        self.error_metrics = error_metrics

        print(error_metrics)

    def __str__(self):
        return f"Epoch object with {len(self.example_evals)} examples\nError metrics: {json.dumps(self.error_metrics, indent=4)}"
    
    def view_results(self):
        '''
        View the results (categorized by matching, missing, extra, and miscategorized) for all examples.
        '''
        for i in range(len(self.example_evals)):
            print(f"Example {i}")
            print(self.example_evals[i].view_results() + '\n')

    def get_error_metrics(self, example_eval_objs):
        # Get error metrics for all examples
        error_metrics = {
            "percent matching results": 0,
            "percent missing results": 0,
            "percent extra results": 0,
            "percent miscategorized results": 0,
            "percent errors": 0
        }

        num_label_results = 0
        num_matching_results = 0
        num_missing_results = 0
        num_extra_results = 0
        num_miscategorized_results = 0
        num_errors = 0
        for example_eval_obj in example_eval_objs:
            num_label_results += len(example_eval_obj.labels)
            num_matching_results += len(example_eval_obj.matchingResults)
            num_missing_results += len(example_eval_obj.missingResults)
            num_extra_results += len(example_eval_obj.extraResults)
            num_miscategorized_results += len(example_eval_obj.miscategorizedResults)
            num_errors += len(example_eval_obj.errors)
        
        # Get error metrics. Found results, missing results, extra results are all divided by the number of label results.
        # Miscategorized results are divided by all results w the same name.
        error_metrics["percent matching results"] = num_matching_results / num_label_results
        error_metrics["percent missing results"] = num_missing_results / num_label_results
        error_metrics["percent extra results"] = num_extra_results / num_label_results
        error_metrics["percent miscategorized results"] = num_miscategorized_results / (num_matching_results + num_miscategorized_results)
        error_metrics["percent errors"] = num_errors / num_label_results

        return error_metrics

    def link_results(self, example_num, linked_names):
        self.example_evals[example_num].link_results(linked_names)
        self.error_metrics = self.get_error_metrics(self.example_evals)
    
class ExampleEvalObject:
    def __init__(self, example_eval):
        self.predictions = example_eval["predictions"]
        self.labels = example_eval["labels"]
        self.errors = example_eval["errors"]
        self.matchingResults = []
        self.miscategorizedResults = []
        self.missingResults = []
        self.extraResults = []

        # Try to directly match names
        found_pred_results = []
        found_label_results = []
        for pred_i, pred_result in enumerate(example_eval["predictions"]):
            for lab_i, label_result in enumerate(example_eval["labels"]):
                if pred_result["name"] == label_result["name"]:
                    if (pred_result["category"] != label_result["category"]):
                        self.miscategorizedResults.append({"prediction": pred_result, "label": label_result})
                    else:
                        self.matchingResults.append({"prediction": pred_result, "label": label_result})
                    found_pred_results.append(pred_i)
                    found_label_results.append(lab_i)
                    break

        # Set missing results to labels that were not found
        for lab_i, label_result in enumerate(example_eval["labels"]):
            if lab_i not in found_label_results:
                print(label_result)
                self.missingResults.append(label_result)

        # Set extra results to predictions that were not found
        for pred_i, pred_result in enumerate(example_eval["predictions"]):
            if pred_i not in found_pred_results:
                self.extraResults.append(pred_result)
    
    # def __str__(self):
    #     return ("Example eval object\n"+
    #         f"Errors:\n{'\n'.join(self.errors)}\n"+
    #         f"Missing results: {[result['name'] for result in self.missingResults]}\n"+
    #         f"Extra results: {[result['name'] for result in self.extraResults]}\n"+
    #         f"Matching results: {[[result['prediction'], result['label']] for result in self.matchingResults]}")
    
    def view_results(self):
        return(
            f"Errors: {json.dumps(self.errors, indent=4)}\n"+
            f"Missing results: {[result['name'] for result in self.missingResults]}\n"+
            f"Extra results: {[result['name'] for result in self.extraResults]}\n"+
            f"Matching results: {[[result['prediction']['name'], result['label']['name']] for result in self.matchingResults]}\n"+
            f"Miscategorized results: {[[result['prediction'], result['label']] for result in self.miscategorizedResults]}"
        )

    def link_results(self, linked_names):
        for i in range(len(linked_names)):
            print(linked_names[i])
            pred_result = [result for result in self.extraResults if any([name == result["name"] for name in linked_names[i]])][0]
            self.extraResults = [result for result in self.extraResults if not any([name == result["name"] for name in linked_names[i]])]
            label_result = [result for result in self.missingResults if any([name == result["name"] for name in linked_names[i]])][0]
            self.missingResults = [result for result in self.missingResults if not any([name == result["name"] for name in linked_names[i]])]
            if (pred_result["category"] != label_result["category"]):
                self.miscategorizedResults.append({"prediction": pred_result, "label": label_result})
            else:
                self.matchingResults.append({"prediction": pred_result, "label": label_result})
       
