library(dplyr)

# Binary variables 
BooleanPrimitive <- c(0, 1) 

# Vegetation height: # Every integer value from 0 to 50 metres 
VegetationHeightPrimitive <- 0:50 

# Vegetation canopy cover:
CanopyCoverPrimitive <- (0:100) / 100 

# Elevation: 
ElevationPrimitive <- 0:5000 

# Categorical variable 
PlantSpreadingPrimitive <- c( "unevenlySpread", "regularRows")

# save all primitive value type in a list
Variables <- list(
  
  waterbody_dominance = BooleanPrimitive,
  aquaculture = BooleanPrimitive,
  builtup_dominance = BooleanPrimitive,
  bareSoil_dominance = BooleanPrimitive,
  mining = BooleanPrimitive,
  herb_dominance = BooleanPrimitive,
  palm_dominance = BooleanPrimitive,
  tree_dominance = BooleanPrimitive,
  
  vegetation_height = VegetationHeightPrimitive,
  
  agricultural_activity = BooleanPrimitive,
  mangrove_presence = BooleanPrimitive,
  rubber_presence = BooleanPrimitive,
  other_species_presence = BooleanPrimitive,
  coffee_presence = BooleanPrimitive,
  cacao_presence = BooleanPrimitive,
  oilpalm_presence = BooleanPrimitive,
  coconut_presence = BooleanPrimitive,
  paddy_presence = BooleanPrimitive,
  timber_extraction = BooleanPrimitive,
  
  vegetation_canopy_cover = CanopyCoverPrimitive,
  
  elevation = ElevationPrimitive,
  
  road_presence = BooleanPrimitive,
  wetland = BooleanPrimitive,
  
  plant_horizonal_spreading = PlantSpreadingPrimitive
)

# generate all possible combination

set.seed(123)

# number of synthetic observations to generate
n <- 750000

# sample independently from each variable's possible values 

SyntheticData <- data.frame( 
  id = seq_len(n), 
  lapply( 
    Variables, 
    function(x) { 
      sample(x, 
             size = n, 
             replace = TRUE) 
      } 
    ), 
  check.names = FALSE )

summary(SyntheticData)

# quality control check 1

# Proportional value first of each element
# generate a fake proportional value from 0 to 100 (divided by 100 later) for each object ending with _dominance (example, waterbody_proportion)
# all primitives ending with _proportion cannot sum more than 1
# if there are primitives that has value more than 0.5, then the corresponding _dominance should be assigned to 1
# in theory, there should only be one _dominance primitive with value of 1
set.seed(123)
SyntheticData <- SyntheticData %>%
  mutate(
    tree_proportion = runif(n()),
    palm_proportion = runif(n()),
    herb_proportion = runif(n()),
    waterbody_proportion = runif(n()),
    builtup_proportion = runif(n()),
    bareSoil_proportion = runif(n())
  )
SyntheticData <- SyntheticData %>%
  mutate(
    proportion_sum = tree_proportion + palm_proportion + herb_proportion + 
      waterbody_proportion + builtup_proportion + bareSoil_proportion,
    tree_proportion = tree_proportion / proportion_sum,
    palm_proportion = palm_proportion / proportion_sum,
    herb_proportion = herb_proportion / proportion_sum,
    waterbody_proportion = waterbody_proportion / proportion_sum,
    builtup_proportion = builtup_proportion / proportion_sum,
    bareSoil_proportion = bareSoil_proportion / proportion_sum,
    proportion_sum = NULL  # Remove helper column
  )

# Assign dominance (value = 1) only to the vegetation type with highest proportion
SyntheticData <- SyntheticData %>%
  mutate(
    # Reset all dominance variables to 0
    tree_dominance = 0,
    palm_dominance = 0,
    herb_dominance = 0,
    waterbody_dominance = 0,
    builtup_dominance = 0,
    bareSoil_dominance = 0,
    
    # Set dominance = 1 only if proportion > 0.5
    tree_dominance = ifelse(tree_proportion > 0.5, 1, 0),
    palm_dominance = ifelse(palm_proportion > 0.5, 1, 0),
    herb_dominance = ifelse(herb_proportion > 0.5, 1, 0),
    waterbody_dominance = ifelse(waterbody_proportion > 0.5, 1, 0),
    builtup_dominance = ifelse(builtup_proportion > 0.5, 1, 0),
    bareSoil_dominance = ifelse(bareSoil_proportion > 0.5, 1, 0)
  )


# Verify QC1: Check that exactly one dominance = 1 per row
dominance_cols <- c("tree_dominance", "palm_dominance", "herb_dominance", 
                    "waterbody_dominance", "builtup_dominance", "bareSoil_dominance")
dominance_sum <- rowSums(SyntheticData[, dominance_cols])

if (all(dominance_sum == 1)) {
  print(" QC1 PASSED: Exactly one dominance variable = 1 per row")
} else {
  print(paste(" QC1 FAILED: Found", sum(dominance_sum != 1), "rows with incorrect dominance count"))
}

# insert code here

# quality control check 2

# remove impossible combination (independent of definition, just physically impossible)
# 1.if tree_proportion AND palm_proportion AND herb_proportion are 0, then vegetation_height should also be 0

SyntheticData <- SyntheticData %>%
  mutate(
    vegetation_height = case_when(
      tree_proportion == 0 & palm_proportion == 0 & herb_proportion == 0 ~ 0,
      TRUE ~ vegetation_height
    )
  )

# 2: If vegetation_height = 0, then vegetation_canopy_cover should be 0
SyntheticData <- SyntheticData %>%
  mutate(
    vegetation_canopy_cover = case_when(
      vegetation_height == 0 ~ 0,
      TRUE ~ vegetation_canopy_cover
    )
  )

write.csv(SyntheticData, "synthetic_monte_carlo_v2.csv", row.names = FALSE )
