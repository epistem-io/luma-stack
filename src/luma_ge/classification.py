import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import ee
from .ee_config import ensure_ee_initialized

# Do not initialize Earth Engine at import time. Initialize when classes are instantiated.

# Module 6: Land Cover Classification
## System Response 6.2 Classification
class FeatureExtraction:
    """
    Perform feature extraction as one of the input for land cover classification. Three types of split is presented here:
    Random Split: splitting the input data randomly based on specified split ratio 
    """
    def __init__(self):
        """
        Initializing the class function for feature extraction
        Ensure Earth Engine is initialized lazily (avoids import-time failures).
        """
        # Ensure Earth Engine is initialized when first used (raises helpful error if not)
        ensure_ee_initialized()
    ############################# 1. Single Random Split ###########################
    #extract pixel value for the labeled region of interest and partitioned them into training and testing data
    #This can be used if the training/reference data is balanced across class and required more fast result
    def random_split(self, image, roi, class_property, split_ratio = 0.6, pixel_size = 30, tile_scale=16):
        """
        Perform single random split and extract pixel value from the imagery
            Parameters:
                image = ee.Image
                aoi = area of interest, ee.FeatureCollection
                split_ratio = 
            Returns:
                tuple: (training_samples, testing_samples)
        """
        #create a random column
        roi_random = roi.randomColumn()
        #partioned the original training data
        training = roi_random.filter(ee.Filter.lt('random', split_ratio))
        testing = roi_random.filter(ee.Filter.gte('random', split_ratio))
        #extract the pixel values
        training_pixels = image.sampleRegions(
                            collection=training,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        testing_pixels = image.sampleRegions(
                            collection=testing,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        print('Single Random Split Training Pixel Size:', training_pixels.size().getInfo())
        print('Single Random Split Testing Pixel Size:', testing_pixels.size().getInfo())
        return training_pixels, testing_pixels
    ############################## 2. Strafied Random Split ###########################
    # Conduct stratified train and test split, ideal for proportional split of the data
    def stratified_split (self, roi, image, class_prop, pixel_size= 30, train_ratio = 0.7, seed=0):
        """
        Used stratified random split to partitioned the original sample data into training and testing data used for model development
        Args:
            Split the region of interest using a stratified random approach, which use class label as basis for splitting
            roi: ee.FeatureCollection (original region of interest)
            class_prop: Class property (column) contain unique class ID
            tran_ratio: ratio for train-test split (usually 70% for training and 50% for testing)
        Return:
        ee.FeatureCollection, consist of training and testing data
        
        """
        #Define the unique class id using aggregate array
        classes = roi.aggregate_array(class_prop).distinct()
        #split the region of interest based on the class
        def split_class (c):
            subset = (roi.filter(ee.Filter.eq(class_prop, c))
                    .randomColumn('random', seed=seed))
            train = (subset.filter(ee.Filter.lt('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'training')))
            test = (subset.filter(ee.Filter.gte('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'testing')))
            return train.merge(test)
        #map the function for all the class
        split_fc = ee.FeatureCollection(classes.map(split_class)).flatten()
        #filter for training and testing
        train_fc = split_fc.filter(ee.Filter.eq('fraction', 'training'))
        test_fc = split_fc.filter(ee.Filter.eq('fraction', 'testing'))
        print('Stratified Random Split Training Pixel Size:', train_fc.size().getInfo())
        print('Stratified Random Split Testing Pixel Size:', test_fc.size().getInfo())      
        #sample the image based stratified split data
        train_pix = image.sampleRegions(
                            collection=train_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16)
        test_pix = image.sampleRegions(
                            collection = test_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16
        )
  
        return train_pix, test_pix

class Generate_LULC:
    def __init__(self):
        """
        Initialize the classification class to
        Perform classification to generate Land Cover Land Use Map. The parameters used in the classification should be the result of hyperparameter tuning
        """

        # Registry of prebuilt wall-to-wall maps per default scheme.
        # Add new years here as maps are finalized — nothing else needs to change.
        self._prebuilt_registry = {
            "RESTORE+ Project": {
                "assets": {
                    2018: "users/hadicu06/IIASA/RESTORE/classified_maps/class_hardened/with_crowdsourced_data/primary_classification" #PLACEHOLDER ASSET DIRECTORY
                },
                "class_band": "classification",
                "scale": 100,
            }
        }

    ############################# 1. Multiclass Classification ###########################
    def hard_classification(self, training_data, class_property, image, ntrees = 100, 
                                  v_split = None, min_leaf = 1, return_model = False,  seed=0):
        """
        Perform multiclass hard classification to generate land cover land use map
            Parameters:
            training data: ee.FeatureCollection, input sample data from feature extraction function (must contain pixel value)
            class_property (str): Column name contain land cover class id
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
        returns:
        ee.Image contain hard multiclass classification
        """
   # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used square root of total bands used in the classification
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        #Random Forest initialization
        clf = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed)
        model = clf.train(
            features=training_data,
            classProperty=class_property,
            inputProperties=image.bandNames()
        )
        #Implement the trained model to classify the whole imagery
        multiclass = image.classify(model)
        #Return model and classification, used for classification summary
        if return_model:
            return multiclass, model
        #if not needed, only return classification result
        else:
            return multiclass   
    ############################# 1. One-vs-rest (OVR) binary Classification ###########################
    def soft_classification(self, training_data, class_property, image, include_final_map=True,
                                ntrees = 100, v_split = None, min_leaf=1, seed=0, probability_scale = 100):
        """
        Implementation of one-vs-rest binary classification approach for multi-class land cover classification, similar to the work of
        Saah et al 2020. This function create probability layer stack for each land cover class. The final land cover map is created using
        maximum probability, via Argmax

        Parameters
            training_data (ee.FeatureCollection): The data which already have a pixel value from input covariates
            class_property (str): Column name contain land cover class id
            image (ee.Image): Image data
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
            probability scale = used to scaled up the probability layer

        Returns:
            ee.Image: Stacked probability bands + final classified map.
        """
        # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used 
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        
        # Get distinct classes ID from the training data. It should be noted that unique ID should be in integer, since 
        # float types tend to resulted in error during the band naming process 
        class_list = training_data.aggregate_array(class_property).distinct()
        
        #Define how to train one vs rest classification and map them all across the class
        def per_class(class_id):
            class_id = ee.Number(class_id)
            #Creating a binary features, 1 for a certain class and 0 for other (forest = 1, other = 0)
            binary_train = training_data.map(lambda ft: ft.set('binary', ee.Algorithms.If(
                            ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                )
                            ))
            #Build random forest classifiers, setting the outputmode to 'probability'. The probability mode will resulted in
            #one binary classification for each class. This give flexibility in modifying the final weight for the final land cover
            #multiprobability resulted in less flexibility in modifying the class weight
            #(the parameters required tuning)
            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed
            ).setOutputMode("PROBABILITY")
            #Train the model
            trained = classifier.train(
                features=binary_train,
                classProperty="binary",
                inputProperties=image.bandNames()
            )
            # Apply to the image and get the probability layer
            # (probability 1 represent the confidence of a pixel belonging to target class)
            prob_img = image.classify(trained).multiply(probability_scale).round().byte()
            #rename the bands
            #Ensure class_id is integer. 
            class_id_str = class_id.int().format()
            band_name = ee.String ('prob_').cat(class_id_str)

            return prob_img.rename(band_name)
        # Map over classes to get probability bands
        prob_imgs = class_list.map(per_class)
        prob_imgcol = ee.ImageCollection(prob_imgs)
        prob_stack = prob_imgcol.toBands()

        #if final map  is not needed, the functin will return prob bands only
        if not include_final_map:
            return prob_stack
        #final map creation using argmax
        print('Creating final classification using argmax')
        class_ids = ee.List(class_list)
        #find the mad prob in each band for each pixel
        #use an index image (0-based) indicating which class has highest probability
        max_prob_index = prob_stack.toArray().arrayArgmax().arrayGet(0)

        #map the index to actual ID
        final_lc = max_prob_index.remap(ee.List.sequence(0, class_ids.size().subtract(1)),
                                        class_ids).rename('classification')
        #calculate confidence layer
        max_confidence = prob_stack.toArray().arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename('confidence')
        #stack the final map and confidence
        stacked = prob_stack.addBands([final_lc, max_confidence])
        return stacked
        ############################# Feature importance ###########################
        #feature importance that can be used by hard or soft classification
    ## System Response 6.3 Model Evaluation
    def get_feature_importance(self, trained_model, training_data=None, class_property=None):
        """
        Extract feature importance from a trained Random Forest model
        Parameters:
            trained_model: ee.Classifier - Trained Random Forest model
            training_data: ee.FeatureCollection - Training data used to train the model (optional, for fallback)
            class_property: str - Class property name (optional, for fallback)
        Returns:
            pandas.DataFrame containing model's feature importance (unitless values)
        """
        try:
            # Try to get model explanation directly
            model_explanation = trained_model.explain().getInfo()
            
            # Extract feature importance
            if 'importance' not in model_explanation:
                raise ValueError("Feature importance not available in model explanation")
            
            importance_dict = model_explanation['importance']
            
            # Create DataFrame containing the importance
            importance_df = pd.DataFrame([
                    {'Band': band, 'Importance': importance}
                    for band, importance in importance_dict.items()
                ]).sort_values('Importance', ascending=False)
            
            # Reset index
            importance_df = importance_df.reset_index(drop=True)
            return importance_df
            
        except Exception as e:
            # If direct explanation fails, try alternative approach
            if training_data is not None and class_property is not None:
                try:
                    # Get band names from training data
                    sample_feature = training_data.first()
                    band_names = sample_feature.propertyNames().filter(ee.Filter.neq('item', class_property)).getInfo()
                    
                    # Create a simple importance estimate based on model structure
                    # This is a fallback - not as accurate as true feature importance
                    num_bands = len(band_names)
                    
                    # Create placeholder importance (equal weights as fallback)
                    importance_df = pd.DataFrame([
                        {'Band': band, 'Importance': 1.0/num_bands}
                        for band in band_names
                    ]).sort_values('Band')
                    
                    print("Warning: Using fallback feature importance (equal weights). True feature importance not available.")
                    return importance_df
                    
                except Exception as fallback_error:
                    raise ValueError(f"Could not extract feature importance. Original error: {str(e)}. Fallback error: {str(fallback_error)}")
            else:
                raise ValueError(f"Could not extract feature importance: {str(e)}. Try providing training_data and class_property for fallback method.") 
    
    def evaluate_model(self, trained_model, test_data, class_property):
        """
        Perform model evaluation based on confusion matrix. This approach is similar to standard Remote Sensing accuracy assessment, but applied for trained model (ee.classifier)
        instead on classification result or map.
        Parameters:
            trained_model: ee.Classifier - Trained Random Forest model
            test_data: ee.FeatureCollection - Testing samples with pixel values
            class_property (str): Column name containing class labels
        
        Returns:
            dict: Dictionary containing accuracy metrics
        """
        #Classify the testing data
        test_classified = test_data.classify(trained_model)
        
        #Get confusion matrix
        confusion_matrix = test_classified.errorMatrix(
            actual=class_property,
            predicted='classification'
        )
        
        # Get the actual class IDs that appear in the confusion matrix
        # These are the classes that were actually present in the test data and predicted by the model
        actual_class_ids = test_data.aggregate_array(class_property).distinct().sort().getInfo()
        predicted_class_ids = test_classified.aggregate_array('classification').distinct().sort().getInfo()
        
        #Get accuracy metrics
        #Producer accuracy / Recall (sensitivity)
        #User accuracy / Precision 
        # Get accuracy metrics from confusion matrix object
        overall_accuracy = confusion_matrix.accuracy().getInfo()
        kappa = confusion_matrix.kappa().getInfo()
        #here still used earth engine terminology
        producers_accuracy_ls = confusion_matrix.producersAccuracy().getInfo()
        #here still used earth engine terminology
        consumers_accuracy_ls = confusion_matrix.consumersAccuracy().getInfo()
        confusion_matrix_array = confusion_matrix.getInfo()

        #Flatten using numpy
        producers_accuracy = np.array(producers_accuracy_ls).flatten().tolist()
        consumers_accuracy = np.array(consumers_accuracy_ls).flatten().tolist()
        
        # Calculate F1 scores and geometric mean score
        #create and empty dict for storing the result
        #change the terminology from earth engine to machine learning
        f1_scores = []
        gmean_per_class = []
        for i in range(len(producers_accuracy)):
            recall_acc = producers_accuracy[i] #recall (machine learning terms)
            precision_acc = consumers_accuracy[i] #precision (machine learning terms)
            #calculate each class f1 score first
            if recall_acc + precision_acc > 0:
                f1 = 2 * (recall_acc * precision_acc) / (recall_acc + precision_acc)
            
            else:
                f1 = 0
            f1_scores.append(f1)
            
            #geometric mearn per class
            #equation: sqrt(recall * precision)
            if recall_acc > 0 and precision_acc > 0:
                gmean = np.sqrt(recall_acc * precision_acc)
            else:
                gmean = 0
            gmean_per_class.append(gmean)
        
        #Overall gmean
        #calculate the overall value of gmean from each class gmean
        #log transform are used for numerical stability (exp(mean(log(r1), log(r2), .... log(rn))))
        valid_gmeans = [g for g in gmean_per_class if g > 0]
        if valid_gmeans:
            overall_gmean = np.exp(np.mean(np.log(valid_gmeans)))
        else:
            overall_gmean = 0

        #Compile the accuracy metrics results
        accuracy_metrics = {
            'overall_accuracy': overall_accuracy,
            'kappa': kappa,
            'precision': consumers_accuracy,
            'recall': producers_accuracy,
            'confusion_matrix': confusion_matrix_array,
            'actual_class_ids': actual_class_ids,
            'predicted_class_ids': predicted_class_ids,
            'f1_scores': f1_scores,
            'gmean_per_class': gmean_per_class,
            'overall_gmean': overall_gmean
        }
        
        return accuracy_metrics
    
    def reclassify_map_by_classes(
        self,
        classification_map: ee.Image,
        classification_df,
        selected_classes: Dict[str, Any],
        other_class_id: int = 999,
        scale: int = 30
    ) -> Tuple[ee.Image, Dict[str, Any]]:
        """
        Reclassify a land cover map based on selected classes of interest.

        Parameters
        ----------
        classification_map : ee.Image
            Original classified map.
        classification_df : pandas.DataFrame
            DataFrame containing classification scheme with 'ID' column.
        selected_classes : Dict[str, Any]
            Dictionary containing selected classes (output from store_classes_of_interest).
        other_class_id : int, default=999
            ID assigned to all non-selected classes.
        scale : int, default=30
            Spatial resolution used for histogram check.

        Returns
        -------
        Tuple[ee.Image, Dict[str, Any]]
            - Reclassified Earth Engine image
            - Metadata dictionary with histogram and validation info
        """

        # List of all class IDs in scheme
        all_class_ids = classification_df["ID"].tolist()

        # Selected classes
        classes_of_interest = selected_classes["classes_of_interest"]

        # Build remap values
        to_values = [
            cid if cid in classes_of_interest else other_class_id
            for cid in all_class_ids
        ]

        # Apply remap
        final_map = classification_map.remap(
            all_class_ids,
            to_values,
            other_class_id
        ).toInt()

        # ---------------------------------
        # Check class presence in AOI
        # ---------------------------------
        histogram_dict = final_map.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=classification_map.geometry(),
            scale=scale,
            maxPixels=1e13
        ).getInfo()

        band_name = list(histogram_dict.keys())[0]
        histogram = histogram_dict[band_name]

        existing_classes = [c for c in classes_of_interest if str(c) in histogram]
        missing_classes = [c for c in classes_of_interest if str(c) not in histogram]

        # Guardrail: show warning message if none of the selected classes are present in the final map
        if len(existing_classes) == 0:
            raise RuntimeError(
            f"None of the selected classes of interest were found in the final classification map. No class were reclassified."
            f"Requested classes: {classes_of_interest}"
        )

        metadata = {
            "histogram": histogram,
            "existing_classes": existing_classes,
            "missing_classes": missing_classes,
            "all_classes_present": len(missing_classes) == 0
        }

        return final_map, metadata
    
    def classify_from_prebuilt(
        self,
        scheme_name: str,
        aoi: ee.Geometry,
        year: int,
        scheme_classes: List[Dict[str, Any]],  # manager.classes passed in directly
    ) -> Dict[str, Any]:
        """
        Bypass the full RF pipeline and return a clipped prebuilt wall-to-wall
        map when the user selects a default scheme.

        Replaces hard_classification() / soft_classification() when a prebuilt
        map is available. The return dict mirrors what the RF path produces so
        downstream code (display, export, legend) needs no branching.

        Parameters
        ----------
        scheme_name    : str               — must match a key in _prebuilt_registry
        aoi            : ee.Geometry       — user's area of interest
        year           : int               — user's temporal input
        scheme_classes : List[Dict]        — manager.classes (ID / Class Name / Color Code)

        Returns
        -------
        Dict with keys:
            final_map        : ee.Image        — clipped classified map
            present_classes  : List[Dict]      — scheme_classes filtered to AOI
            vis_params       : dict            — ee-style vis params
            source           : str             — "prebuilt_bypass"
            year_used        : int             — actual year loaded (may differ if fallback)
            scheme           : str             — scheme_name
        """
        cfg = self._prebuilt_registry.get(scheme_name)
        if cfg is None:
            raise ValueError(
                f"No prebuilt map registered for scheme '{scheme_name}'. "
                f"Available: {list(self._prebuilt_registry.keys())}"
            )

        # Resolve year with graceful fallback 
        available_years = sorted(cfg["assets"].keys())
        year_used = year

        if year not in cfg["assets"]:
            year_used = min(available_years, key=lambda y: abs(y - year))
            print(
                f"[classify_from_prebuilt] Year {year} not available for '{scheme_name}'. "
                f"Falling back to nearest: {year_used}. Available: {available_years}"
            )

        # Load & clip 
        national_map = ee.Image(cfg["assets"][year_used])
        clipped_map  = national_map.clip(aoi)

        # Detect class IDs present in AOI
        class_band = cfg["class_band"]
        hist_dict = (
            clipped_map
            .select(class_band)
            .reduceRegion(
                reducer   = ee.Reducer.frequencyHistogram(),
                geometry  = aoi,
                scale     = cfg["scale"],
                maxPixels = 1e13,
            )
            .get(class_band)
            .getInfo()                          # {str(class_id): pixel_count}
        )
        present_ids = {int(k) for k, v in hist_dict.items() if v > 0}

        # Filter legend to AOI-present classes 
        # Reuses the exact names & colors already loaded in manager.classes,
        # so Module 2 legend and final map legend are always consistent
        present_classes = [c for c in scheme_classes if c['ID'] in present_ids]

        # ── 5. Build vis params ───────────────────────────────────────────────
        ids     = [c['ID']         for c in present_classes]
        palette = [c['Color Code'] for c in present_classes]
        labels  = [c['Class Name'] for c in present_classes]

        vis_params = {
            "bands":       [class_band],
            "min":         min(ids) if ids else 0,
            "max":         max(ids) if ids else 1,
            "palette":     palette,
            "class_names": labels,
        }

        return {
            "final_map":       clipped_map,
            "present_classes": present_classes,
            "vis_params":      vis_params,
            "source":          "prebuilt_bypass",
            "year_used":       year_used,
            "scheme":          scheme_name,
        }