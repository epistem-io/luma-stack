"""
Module 6: Classification

Provides:
- Hard direct classification (`hard_classification`) and soft probabilistic classification
- Support feature extraction, model training, and model evaluation
"""
import logging
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Sequence, cast
import ee
from .ee_config import ensure_ee_initialized

logger = logging.getLogger(__name__)

#return class label name from scheme (module 2)
def extract_class_name_map(class_names: Any) -> Dict[int, str]:
    """Normalize a dict, DataFrame scheme, or class-dict list into ``{class_id: class_name}``."""
    if class_names is None:
        return {}
    if isinstance(class_names, dict):
        return {int(k): str(v).strip() for k, v in class_names.items() if v is not None}
    if isinstance(class_names, pd.DataFrame):
        id_col = next((c for c in ('ID', 'id', 'class_id', 'ID_Kelas') if c in class_names.columns), None)
        name_col = next(
            (
                c
                for c in ('Class Name', 'Land Cover Class', 'LULC_Type', 'class_name', 'name', 'KelasTupla')
                if c in class_names.columns
            ),
            None,
        )
        if id_col is None or name_col is None:
            return {}
        out: Dict[int, str] = {}
        for _, row in class_names.iterrows():
            if pd.notna(row[id_col]) and pd.notna(row[name_col]):
                out[int(row[id_col])] = str(row[name_col]).strip()
        return out
    if isinstance(class_names, (list, tuple)):
        out = {}
        for item in class_names:
            if isinstance(item, dict) and 'ID' in item:
                nm = item.get('Class Name') or item.get('Land Cover Class') or item.get('LULC_Type')
                if nm is not None:
                    out[int(item['ID'])] = str(nm).strip()
        return out
    return {}

#build probability band names from class scheme (module 2)
def build_prob_band_names(
    class_ids: Sequence[int],
    class_names: Any = None,
    prefix: str = 'prob_',
) -> Tuple[List[str], Dict[int, str]]:
    """
    Build Earth Engine-safe probability band names and ``{class_id: band_name}`` map.

    When ``class_names`` is omitted (or lacks an ID), defaults to ``prob_<id>``.
    When provided, sanitizes scheme class names to ``[A-Za-z0-9_]+`` (e.g.
    ``prob_Hutan_Lahan_Kering``) and guarantees uniqueness across classes.
    """
    name_map = extract_class_name_map(class_names)
    band_names: List[str] = []
    cid_to_band: Dict[int, str] = {}
    seen = set()

    for raw_cid in class_ids:
        cid = int(raw_cid)
        raw_name = name_map.get(cid)
        if raw_name:
            sanitized = re.sub(r'[^A-Za-z0-9]+', '_', raw_name).strip('_')
            candidate = f"{prefix}{sanitized}" if sanitized else f"{prefix}{cid}"
        else:
            candidate = f"{prefix}{cid}"

        if candidate in seen:
            candidate = f"{candidate}_{cid}"
        seen.add(candidate)
        band_names.append(candidate)
        cid_to_band[cid] = candidate

    return band_names, cid_to_band

#compuete cross entropy loss from predicted probability 
def compute_log_loss_from_arrays(
    y_true: Sequence[int],
    y_prob: Any,
    class_ids: Optional[Sequence[int]] = None,
    eps: float = 1e-6,
) -> float:
    """
    Compute multi-class log loss (cross-entropy) from predicted probability vectors.

    Parameters
    ----------
    y_true : sequence of int
        Ground truth class IDs of length N.
    y_prob : array-like of shape (N, K)
        Predicted class probabilities (each row ordered along ``class_ids``).
        If values appear scaled (row sums > 1.5), rows are normalized by row sum.
    class_ids : sequence of int, optional
        Ordered class IDs matching columns of ``y_prob``. Defaults to ``sorted(set(y_true))``.
    eps : float, default=1e-6
        Probability clipping threshold to avoid ``log(0)``.
    """
    y_true_arr = [int(v) for v in y_true]
    probs = np.asarray(y_prob, dtype=np.float64)
    if probs.ndim != 2 or len(y_true_arr) != probs.shape[0]:
        raise ValueError(
            f"y_prob shape {probs.shape} does not match len(y_true)={len(y_true_arr)}."
        )
    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute log loss on an empty sample set.")

    ids = [int(c) for c in (class_ids if class_ids is not None else sorted(set(y_true_arr)))]
    if len(ids) != probs.shape[1]:
        raise ValueError(
            f"len(class_ids)={len(ids)} does not match probability columns {probs.shape[1]}."
        )

    id_to_col = {cid: idx for idx, cid in enumerate(ids)}
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums > 1.5):
        probs = np.divide(probs, np.where(row_sums > 0, row_sums, 1.0))

    total_nll = 0.0
    for row_idx, cid in enumerate(y_true_arr):
        if cid not in id_to_col:
            raise ValueError(f"Ground truth class ID {cid} missing from class_ids {ids}.")
        p = float(np.clip(probs[row_idx, id_to_col[cid]], eps, 1.0 - eps))
        total_nll -= float(np.log(p))
    return total_nll / float(len(y_true_arr))

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
    def stratified_split (self, roi, image, class_prop, pixel_size= 30, train_ratio = 0.7, seed=0, tile_scale=16):
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
                            tileScale = tile_scale)
        test_pix = image.sampleRegions(
                            collection = test_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = tile_scale
        )
  
        return train_pix, test_pix

    ############################## 3. Multitemporal Feature Extraction ###########################
    def extract_multitemporal_samples(
        self,
        predictor_stacks: Dict[str, ee.Image],
        training_splits: Dict[str, Any],
        class_property: str = 'class_id',
        reference_period: str = 'baseline',
        common_bands: Optional[List[str]] = None,
        mode: str = 'auto',
        scale: int = 30,
        tile_scale: int = 4,
        split_method: str = 'stratified',
        train_ratio: float = 0.7,
        seed: int = 0,
        period_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract pixel values (`sampleRegions`) from multitemporal predictor stacks.

        Routes automatically based on the supplied training datasets:
        - **1 dataset (only ``reference_period`` / most recent imagery)**: executes
          **single feature extraction** on ``reference_period`` only
          (``mode_used='model_transfer'``, ``extraction_type='single'``), aligning
          to ``common_bands`` when provided.
        - **$N$ datasets (one per period)**: executes **batch feature extraction**
          independently across every period (``mode_used='batch'``,
          ``extraction_type='batch'``).

        Each period entry in ``training_splits`` may be:
        - An unsplit ``ee.FeatureCollection`` or ``GeoDataFrame`` (split internally via
          ``stratified_split`` or ``random_split``);
        - A ``dict`` returned by ``SyncTrainData.LoadMultitemporalTrainData()``
          containing ``'training_data'``;
        - A pre-split ``(train, test)`` tuple/list of ``ee.FeatureCollection`` / ``GeoDataFrame``; or
        - ``None`` (for target periods without reference samples in Model Transfer).
        """
        order = list(period_order) if period_order is not None else list(predictor_stacks.keys())
        if len(order) < 2:
            raise ValueError(
                f"extract_multitemporal_samples requires at least 2 periods, got {len(order)}."
            )

        def _is_non_empty(obj: Any) -> bool:
            if obj is None:
                return False
            if isinstance(obj, dict) and 'training_data' in obj:
                return _is_non_empty(obj['training_data'])
            if isinstance(obj, (tuple, list)):
                return len(obj) > 0 and _is_non_empty(obj[0])
            if isinstance(obj, pd.DataFrame):
                return not obj.empty
            return True

        provided = [p for p in order if _is_non_empty((training_splits or {}).get(p))]

        if mode == 'auto':
            if len(provided) == 1 and provided[0] == reference_period:
                mode_used = 'model_transfer'
            elif len(provided) == len(order):
                mode_used = 'batch'
            else:
                raise ValueError(
                    f"Cannot infer feature extraction mode from training_splits keys {provided} "
                    f"for period_order={order} and reference_period='{reference_period}'."
                )
        elif mode in ('batch', 'model_transfer'):
            mode_used = mode
            if mode_used == 'model_transfer' and reference_period not in provided:
                raise ValueError(
                    f"Model Transfer feature extraction requires training data for reference_period '{reference_period}'."
                )
            if mode_used == 'batch' and len(provided) != len(order):
                missing = [p for p in order if p not in provided]
                raise ValueError(
                    f"Batch feature extraction requires training data for all periods; missing {missing}."
                )
        else:
            raise ValueError(f"mode must be 'auto', 'batch', or 'model_transfer', got '{mode}'")

        extraction_type = 'single' if mode_used == 'model_transfer' else 'batch'
        target_periods = [reference_period] if mode_used == 'model_transfer' else list(order)
        skipped_periods = [p for p in order if p not in target_periods]

        def _to_ee_fc(obj: Any) -> Optional[ee.FeatureCollection]:
            if obj is None:
                return None
            if isinstance(obj, ee.FeatureCollection):
                return obj
            if isinstance(obj, pd.DataFrame):
                if obj.empty:
                    return None
                from .input_utils import EE_converter
                cols = [c for c in (class_property, 'geometry') if c in obj.columns]
                sub_gdf = obj[cols] if 'geometry' in cols and class_property in cols else obj
                return EE_converter(verbose=False).convert_roi_gdf(sub_gdf)
            return obj

        sampled_splits: Dict[str, Tuple[ee.FeatureCollection, Optional[ee.FeatureCollection]]] = {}
        for label in target_periods:
            if label not in predictor_stacks:
                raise ValueError(f"Period '{label}' missing from predictor_stacks.")

            img = predictor_stacks[label]
            if mode_used == 'model_transfer' and common_bands is not None:
                img = img.select(list(common_bands))

            entry = training_splits[label]
            if isinstance(entry, dict) and 'training_data' in entry:
                entry = entry['training_data']

            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                train_fc = _to_ee_fc(entry[0])
                test_fc = _to_ee_fc(entry[1])
                if train_fc is None:
                    raise ValueError(f"Training split for period '{label}' is empty or None.")
                train_pix = img.sampleRegions(
                    collection=train_fc,
                    properties=[class_property],
                    scale=scale,
                    tileScale=tile_scale,
                )
                test_pix = (
                    img.sampleRegions(
                        collection=test_fc,
                        properties=[class_property],
                        scale=scale,
                        tileScale=tile_scale,
                    )
                    if test_fc is not None
                    else None
                )
            else:
                roi_fc = _to_ee_fc(entry)
                if roi_fc is None:
                    raise ValueError(f"Training data for period '{label}' is empty or None.")
                if split_method == 'stratified':
                    train_pix, test_pix = self.stratified_split(
                        roi=roi_fc,
                        image=img,
                        class_prop=class_property,
                        pixel_size=scale,
                        train_ratio=train_ratio,
                        seed=seed,
                        tile_scale=tile_scale,
                    )
                elif split_method == 'random':
                    train_pix, test_pix = self.random_split(
                        image=img,
                        roi=roi_fc,
                        class_property=class_property,
                        split_ratio=train_ratio,
                        pixel_size=scale,
                        tile_scale=tile_scale,
                    )
                else:
                    raise ValueError(
                        f"split_method must be 'stratified' or 'random', got '{split_method}'"
                    )

            sampled_splits[label] = (train_pix, test_pix)

        logger.info(
            "Multitemporal feature extraction (%s / %s): extracted=%s, skipped=%s",
            extraction_type,
            mode_used,
            target_periods,
            skipped_periods,
        )

        return {
            'mode_used': mode_used,
            'extraction_type': extraction_type,
            'sampled_splits': sampled_splits,
            'extracted_periods': target_periods,
            'skipped_periods': skipped_periods,
        }

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
                    2013: "projects/epistem-485512/assets/Indonesia_LULC_2013_100m",
                    2014: "projects/epistem-485512/assets/Indonesia_LULC_2014_100m",
                    2015: "projects/epistem-485512/assets/Indonesia_LULC_2015_100m",
                    2016: "projects/epistem-485512/assets/Indonesia_LULC_2016_100m",
                    2017: "projects/epistem-485512/assets/Indonesia_LULC_2017_100m",
                    2018: "users/hadicu06/IIASA/RESTORE/classified_maps/class_hardened/with_crowdsourced_data/primary_classification", #PLACEHOLDER ASSET DIRECTORY
                    2019: "projects/epistem-490415/assets/RESTORE/Indonesia_Mosaic_LULC_2019_100m",
                    2020: "projects/epistem-490415/assets/RESTORE/Indonesia_Mosaic_LULC_2020_100m",
                    2021: "projects/epistem-490415/assets/RESTORE/Indonesia_Mosaic_LULC_2021_100m",
                    2022: "projects/epistem-490415/assets/RESTORE/Indonesia_Mosaic_LULC_2022_100m",
                    2023: "projects/epistem-490415/assets/RESTORE/Indonesia_Mosaic_LULC_2023_100m",
                    2024: "projects/epistem-485512/assets/Indonesia_LULC_2024_100m",
                    2025: "users/Restore25/with_crowdsourced_data__primary_classification_2025"   
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
    ############################# 1. One-vs-rest (OVR) & Multiprobability Soft Classification ###########################
    def soft_classification(self, training_data, class_property, image, include_final_map=True,
                                ntrees = 100, v_split = None, min_leaf=1, seed=0, probability_scale = 100,
                                mode: str = 'ovr', return_model: bool = False,
                                class_names: Optional[Any] = None,
                                spatial_smoothing: bool = False,
                                smoothing_radius: int = 1,
                                smoothing_sigma: float = 1.0):
        """
        Perform soft (probabilistic) multi-class land cover classification.

        Supports two modes:
        - ``'ovr'`` (default, backwards-compatible): One-vs-Rest binary Random Forest
          classifiers per class (Saah et al., 2020).
        - ``'multiprobability'``: A single Random Forest classifier with
          ``setOutputMode('MULTIPROBABILITY')``, ideal for fast execution and
          cross-period Model Transfer.

        Parameters
        ----------
        training_data : ee.FeatureCollection
            Sample data containing extracted predictor pixel values.
        class_property : str
            Column/property name containing integer land cover class IDs.
        image : ee.Image
            Predictor stack image.
        include_final_map : bool, default=True
            If True, append `'classification'` (argmax) and `'confidence'` (max prob) bands.
        ntrees, v_split, min_leaf, seed :
            Random Forest hyperparameters.
        probability_scale : int, default=100
            Scale multiplier applied before rounding probabilities to byte.
        mode : {'ovr', 'multiprobability'}, default='ovr'
            Soft classification engine.
        return_model : bool, default=False
            When True, returns ``(result_image, model_info)``.

        Returns
        -------
        ee.Image or tuple(ee.Image, dict)
            Stacked probability bands (`prob_<id>`) plus optional `'classification'`
            and `'confidence'` bands.
        """
        # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        if mode not in ('ovr', 'multiprobability'):
            raise ValueError(f"mode must be 'ovr' or 'multiprobability', got '{mode}'")
        #if for some reason var split is not specified, used 
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()

        if mode == 'multiprobability':
            class_ids_raw = training_data.aggregate_array(class_property).distinct().sort().getInfo()
            class_ids = sorted(int(c) for c in class_ids_raw) # type: ignore
            if not class_ids:
                raise ValueError("No distinct classes found in training_data.")

            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees,
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed,
            ).setOutputMode('MULTIPROBABILITY')

            trained_model = classifier.train(
                features=training_data,
                classProperty=class_property,
                inputProperties=image.bandNames(),
            )
            prob_band_names, prob_band_map = build_prob_band_names(class_ids, class_names=class_names)
            stacked = self.classify_with_model(
                trained_model=trained_model,
                image=image,
                mode='soft',
                class_ids=class_ids,
                include_final_map=include_final_map,
                probability_scale=probability_scale,
                class_names=class_names,
                spatial_smoothing=spatial_smoothing,
                smoothing_radius=smoothing_radius,
                smoothing_sigma=smoothing_sigma,
            )
            if return_model:
                return stacked, {
                    'model': trained_model,
                    'class_ids': class_ids,
                    'prob_band_names': prob_band_names,
                    'prob_band_map': prob_band_map,
                    'mode': 'multiprobability',
                }
            return stacked

        # Get distinct classes ID from the training data. It should be noted that unique ID should be in integer, since 
        # float types tend to resulted in error during the band naming process 
        class_list = training_data.aggregate_array(class_property).distinct()
        ovr_name_dict = None
        if class_names is not None:
            parsed_names = extract_class_name_map(class_names)
            if parsed_names:
                _, ovr_prob_map = build_prob_band_names(sorted(parsed_names.keys()), class_names=parsed_names)
                ovr_name_dict = ee.Dictionary({str(k): v for k, v in ovr_prob_map.items()})
        
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
            default_band_name = ee.String('prob_').cat(class_id_str)
            band_name = (
                ee.String(ovr_name_dict.get(class_id_str, default_band_name))
                if ovr_name_dict is not None
                else default_band_name
            )

            return prob_img.rename(band_name)
        # Map over classes to get probability bands
        prob_imgs = class_list.map(per_class)
        prob_imgcol = ee.ImageCollection(prob_imgs)
        prob_stack = prob_imgcol.toBands()
        if spatial_smoothing:
            prob_stack = (
                prob_stack.toFloat()
                .convolve(
                    ee.Kernel.gaussian(
                        radius=smoothing_radius,
                        sigma=smoothing_sigma,
                        units='pixels',
                        normalize=True,
                    )
                )
                .round()
                .byte()
            )

        #if final map  is not needed, the functin will return prob bands only
        if not include_final_map:
            if return_model:
                return prob_stack, {'mode': 'ovr', 'class_list': class_list}
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
        if return_model:
            return stacked, {'mode': 'ovr', 'class_list': class_list}
        return stacked

    def classify_with_model(
        self,
        trained_model: ee.Classifier,
        image: ee.Image,
        mode: str = 'hard',
        class_ids: Optional[List[int]] = None,
        include_final_map: bool = True,
        probability_scale: int = 100,
        class_names: Optional[Any] = None,
        spatial_smoothing: bool = False,
        smoothing_radius: int = 1,
        smoothing_sigma: float = 1.0,
    ) -> ee.Image:
        """
        Apply a previously trained ``ee.Classifier`` (hard or MULTIPROBABILITY)
        to a target ``ee.Image`` predictor stack (used for Model Transfer).
        """
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        if mode not in ('hard', 'soft', 'multiprobability'):
            raise ValueError(f"mode must be 'hard', 'soft', or 'multiprobability', got '{mode}'")

        if mode == 'hard':
            return image.classify(trained_model)

        if not class_ids:
            raise ValueError("class_ids (sorted list of integer class IDs) is required for soft/multiprobability transfer.")

        sorted_ids = sorted(int(c) for c in class_ids)
        prob_band_names, _ = build_prob_band_names(sorted_ids, class_names=class_names)
        raw_prob_img = image.classify(trained_model)
        prob_stack = (
            raw_prob_img
            .arrayFlatten([prob_band_names])
            .multiply(probability_scale)
            .round()
            .byte()
        )
        if spatial_smoothing:
            prob_stack = (
                prob_stack.toFloat()
                .convolve(
                    ee.Kernel.gaussian(
                        radius=smoothing_radius,
                        sigma=smoothing_sigma,
                        units='pixels',
                        normalize=True,
                    )
                )
                .round()
                .byte()
            )
        if not include_final_map:
            return prob_stack

        max_prob_index = prob_stack.toArray().arrayArgmax().arrayGet(0)
        final_lc = (
            max_prob_index
            .remap(list(range(len(sorted_ids))), sorted_ids)
            .rename('classification')
        )
        max_confidence = (
            prob_stack.toArray()
            .arrayReduce(ee.Reducer.max(), [0])
            .arrayGet([0])
            .rename('confidence')
        )
        return prob_stack.addBands([final_lc, max_confidence])
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
    
    def compute_log_loss(
        self,
        trained_model: ee.Classifier,
        test_data: ee.FeatureCollection,
        class_property: str,
        class_ids: Optional[List[int]] = None,
        eps: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Compute optional server-side multi-class Log Loss on a ``MULTIPROBABILITY``
        ``ee.Classifier`` against held-out ``test_data``.

        Evaluates raw probability calibration ($-\frac{1}{N}\sum \log(\max(P(y_i), \epsilon))$)
        without converting probabilities to hard labels, complementing Overall Accuracy
        especially when downstream HMM consistency analysis consumes log-probabilities.
        """
        if not class_ids:
            class_ids = sorted(
                int(c) for c in (test_data.aggregate_array(class_property).distinct().sort().getInfo() or [])
            )
        sorted_ids = sorted(int(c) for c in class_ids)
        sorted_ids_ee = ee.List(sorted_ids)

        valid_test = test_data.filter(ee.Filter.inList(class_property, sorted_ids_ee))
        test_prob_fc = valid_test.classify(trained_model)

        def _feature_nll(ft):
            arr = ee.Array(ft.get('classification'))
            actual = ee.Number(ft.get(class_property)).int()
            idx = sorted_ids_ee.indexOf(actual)
            p_true = ee.Number(arr.get([idx])).max(eps).min(1.0 - eps)
            return ft.set('_nll', p_true.log().multiply(-1.0))

        scored_fc = test_prob_fc.map(_feature_nll)
        stats = scored_fc.aggregate_stats('_nll').getInfo() or {}
        nll_mean = float(stats.get('mean', 0.0) or 0.0)
        n_scored = int(round(float(stats.get('total_count', 0) or 0)))

        return {
            'log_loss': nll_mean,
            'n_samples': n_scored,
            'eps': eps,
            'class_ids': sorted_ids,
        }

    def evaluate_model(
        self,
        trained_model,
        test_data,
        class_property,
        is_multiprobability: bool = False,
        class_ids: Optional[List[int]] = None,
        include_log_loss: bool = False,
        eps: float = 1e-6,
    ):
        """
        Perform model evaluation based on confusion matrix. This approach is similar to standard Remote Sensing accuracy assessment, but applied for trained model (ee.classifier)
        instead on classification result or map.
        Parameters:
            trained_model: ee.Classifier - Trained Random Forest model
            test_data: ee.FeatureCollection - Testing samples with pixel values
            class_property (str): Column name containing class labels
            is_multiprobability (bool): Set True when ``trained_model`` uses MULTIPROBABILITY output mode
            class_ids (list[int], optional): Sorted class IDs matching the MULTIPROBABILITY array order
        
        Returns:
            dict: Dictionary containing accuracy metrics
        """
        #Classify the testing data
        test_classified = test_data.classify(trained_model)
        if is_multiprobability:
            if not class_ids:
                class_ids = sorted(
                    int(c) for c in test_data.aggregate_array(class_property).distinct().sort().getInfo()
                )
            sorted_ids_ee = ee.List(sorted(int(c) for c in class_ids))

            def _argmax_feature(ft):
                arr = ee.Array(ft.get('classification'))
                max_idx = arr.argmax().get(0)
                return ft.set('classification', sorted_ids_ee.get(max_idx))

            test_classified = test_classified.map(_argmax_feature)
        
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

        if include_log_loss and is_multiprobability:
            ll_res = self.compute_log_loss(
                trained_model=trained_model,
                test_data=test_data,
                class_property=class_property,
                class_ids=class_ids,
                eps=eps,
            )
            accuracy_metrics['log_loss'] = ll_res['log_loss']
        
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
            maxPixels=1e13 # type: ignore
        ).getInfo()

        band_name = list(histogram_dict.keys())[0] # type: ignore
        histogram = histogram_dict[band_name] # type: ignore

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
        print("[classify_from_prebuilt] Map resolution (m):", national_map.projection().nominalScale().getInfo())
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

    #migt be removed, not used in the current workflow
    def classify_from_prebuilt_multitemporal(
        self,
        scheme_name: str,
        aoi: ee.Geometry,
        years_by_period: Dict[str, int],
        scheme_classes: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve clipped prebuilt wall-to-wall LULC maps across multiple periods.
        """
        return {
            label: self.classify_from_prebuilt(
                scheme_name=scheme_name,
                aoi=aoi,
                year=year,
                scheme_classes=scheme_classes,
            )
            for label, year in years_by_period.items()
        }
    #core function for multitemporal classification
    def classify_multitemporal(
        self,
        predictor_stacks: Dict[str, ee.Image],
        training_splits: Dict[str, Tuple[ee.FeatureCollection, ee.FeatureCollection]],
        class_property: str = 'class_id',
        mode: str = 'auto',
        reference_period: str = 'baseline',
        classification_type: str = 'hard',
        soft_mode: str = 'multiprobability',
        ntrees: int = 100,
        v_split: Optional[int] = None,
        min_leaf: int = 1,
        seed: int = 0,
        probability_scale: int = 100,
        common_bands: Optional[List[str]] = None,
        period_order: Optional[List[str]] = None,
        class_names: Optional[Any] = None,
        include_log_loss: bool = False,
        spatial_smoothing: bool = False,
        smoothing_radius: int = 1,
        smoothing_sigma: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run multitemporal Random Forest classification across N >= 2 periods in either
        ``'batch'`` mode (one trained model per period) or ``'model_transfer'`` mode
        (model trained on ``reference_period`` and applied to all target periods).

        Supports both ``classification_type='hard'`` and ``classification_type='soft'``
        (with ``soft_mode='multiprobability'`` recommended for fast transferable
        probability stacks).
        """
        if period_order is None:
            period_order = list(predictor_stacks.keys())

        if len(period_order) < 2:
            raise ValueError(f"classify_multitemporal requires at least 2 periods, got {len(period_order)}.")

        if classification_type not in ('hard', 'soft'):
            raise ValueError(f"classification_type must be 'hard' or 'soft', got '{classification_type}'")

        if mode == 'auto':
            provided = [p for p in period_order if p in training_splits and training_splits[p] is not None]
            if len(provided) == 1 and provided[0] == reference_period:
                mode_used = 'model_transfer'
            elif len(provided) == len(period_order):
                mode_used = 'batch'
            else:
                raise ValueError(
                    f"Cannot infer classification mode from training_splits keys {provided} "
                    f"for period_order={period_order} and reference_period='{reference_period}'."
                )
        elif mode in ('batch', 'model_transfer'):
            mode_used = mode
        else:
            raise ValueError(f"mode must be 'auto', 'batch', or 'model_transfer', got '{mode}'")

        maps: Dict[str, ee.Image] = {}
        models: Dict[str, Any] = {}
        feature_importance: Dict[str, pd.DataFrame] = {}
        model_evaluation: Dict[str, Dict[str, Any]] = {}
        class_ids_by_period: Dict[str, List[int]] = {}
        prob_bands_by_period: Dict[str, Dict[int, str]] = {}

        if mode_used == 'batch':
            for label in period_order:
                train_fc, test_fc = training_splits[label]
                stack = predictor_stacks[label]

                if classification_type == 'hard':
                    lulc_img, trained_model = cast(Tuple[ee.Image, Dict[str, Any]], self.hard_classification(
                        training_data=train_fc,
                        class_property=class_property,
                        image=stack,
                        ntrees=ntrees,
                        v_split=v_split,
                        min_leaf=min_leaf,
                        return_model=True,
                        seed=seed,
                    ))
                    is_multiprob = False
                    cids = None
                else:
                    lulc_img, model_info = cast(Tuple[ee.Image, Dict[str, Any]], self.soft_classification(
                        training_data=train_fc,
                        class_property=class_property,
                        image=stack,
                        include_final_map=True,
                        ntrees=ntrees,
                        v_split=v_split,
                        min_leaf=min_leaf,
                        seed=seed,
                        probability_scale=probability_scale,
                        mode=soft_mode,
                        return_model=True,
                        class_names=class_names,
                        spatial_smoothing=spatial_smoothing,
                        smoothing_radius=smoothing_radius,
                        smoothing_sigma=smoothing_sigma,
                    ))
                    trained_model = model_info.get('model')
                    cids = model_info.get('class_ids')
                    is_multiprob = soft_mode == 'multiprobability'
                    if cids is not None:
                        class_ids_by_period[label] = cids
                    if model_info.get('prob_band_map'):
                        prob_bands_by_period[label] = model_info['prob_band_map']

                maps[label] = lulc_img
                models[label] = trained_model
                if trained_model is not None:
                    feature_importance[label] = self.get_feature_importance(
                        trained_model, training_data=train_fc, class_property=class_property
                    )
                    if test_fc is not None:
                        model_evaluation[label] = self.evaluate_model(
                            trained_model=trained_model,
                            test_data=test_fc,
                            class_property=class_property,
                            is_multiprobability=is_multiprob,
                            class_ids=cids,
                            include_log_loss=include_log_loss,
                        )
        else:
            if reference_period not in training_splits:
                raise ValueError(f"training_splits missing reference_period '{reference_period}'.")

            ref_train_fc, ref_test_fc = training_splits[reference_period]
            ref_stack = predictor_stacks[reference_period]
            if common_bands is not None:
                ref_stack = ref_stack.select(common_bands)

            if classification_type == 'hard':
                ref_map, ref_model = cast(Tuple[ee.Image, Dict[str, Any]], self.hard_classification(
                    training_data=ref_train_fc,
                    class_property=class_property,
                    image=ref_stack,
                    ntrees=ntrees,
                    v_split=v_split,
                    min_leaf=min_leaf,
                    return_model=True,
                    seed=seed,
                ))
                is_multiprob = False
                ref_cids = None
                transfer_mode = 'hard'
            else:
                if soft_mode != 'multiprobability':
                    raise ValueError(
                        "Model Transfer with soft classification requires soft_mode='multiprobability' "
                        "so a single trained classifier can be applied across periods."
                    )
                ref_map, model_info = cast(Tuple[ee.Image, Dict[str, Any]],self.soft_classification(
                    training_data=ref_train_fc,
                    class_property=class_property,
                    image=ref_stack,
                    include_final_map=True,
                    ntrees=ntrees,
                    v_split=v_split,
                    min_leaf=min_leaf,
                    seed=seed,
                    probability_scale=probability_scale,
                    mode='multiprobability',
                    return_model=True,
                    class_names=class_names,
                    spatial_smoothing=spatial_smoothing,
                    smoothing_radius=smoothing_radius,
                    smoothing_sigma=smoothing_sigma,
                ))
                ref_model = model_info['model']
                ref_cids = model_info['class_ids']
                is_multiprob = True
                transfer_mode = 'soft'

            maps[reference_period] = ref_map
            models[reference_period] = ref_model
            if ref_cids is not None:
                class_ids_by_period[reference_period] = ref_cids
            if classification_type == 'soft' and model_info.get('prob_band_map'): # type: ignore
                prob_bands_by_period[reference_period] = model_info['prob_band_map'] # type: ignore

            feature_importance[reference_period] = self.get_feature_importance(
                ref_model, training_data=ref_train_fc, class_property=class_property
            )
            if ref_test_fc is not None:
                model_evaluation[reference_period] = self.evaluate_model(
                    trained_model=ref_model,
                    test_data=ref_test_fc,
                    class_property=class_property,
                    is_multiprobability=is_multiprob,
                    class_ids=ref_cids,
                    include_log_loss=include_log_loss,
                )

            for label in period_order:
                if label == reference_period:
                    continue
                target_stack = predictor_stacks[label]
                if common_bands is not None:
                    target_stack = target_stack.select(common_bands)

                maps[label] = self.classify_with_model(
                    trained_model=ref_model, # type: ignore
                    image=target_stack,
                    mode=transfer_mode,
                    class_ids=ref_cids,
                    include_final_map=True,
                    probability_scale=probability_scale,
                    class_names=class_names,
                    spatial_smoothing=spatial_smoothing,
                    smoothing_radius=smoothing_radius,
                    smoothing_sigma=smoothing_sigma,
                )
                models[label] = ref_model
                if ref_cids is not None:
                    class_ids_by_period[label] = ref_cids
                    if classification_type == 'soft':
                        _, pb_map = build_prob_band_names(ref_cids, class_names=class_names)
                        prob_bands_by_period[label] = pb_map

        return {
            'mode_used': mode_used,
            'classification_type': classification_type,
            'soft_mode': soft_mode if classification_type == 'soft' else None,
            'reference_period': reference_period,
            'period_order': period_order,
            'maps': maps,
            'models': models,
            'feature_importance': feature_importance,
            'model_evaluation': model_evaluation,
            'class_ids': class_ids_by_period,
            'prob_bands': prob_bands_by_period,
        }