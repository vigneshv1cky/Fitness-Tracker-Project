library(tidyverse)

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_arc <- read_csv("data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

single_file_gyr <- read_csv("data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files <- list.files(path = "data/raw/MetaMotion", pattern = "*.csv", full.names = TRUE)


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path <- "data/raw/MetaMotion/"
f <- files[1]

participant <- f %>% 
  str_replace(data_path, "") %>% 
  str_split("-") %>% 
  pluck(1, 1)

label <- f %>% 
  str_split("-") %>% 
  pluck(1, 2)

category <- f %>% 
  str_split("-") %>% 
  pluck(1, 3) %>% 
  str_remove("123") %>% 
  str_remove("_MetaWear_2019")

df <- read_csv(f) %>%
  mutate(participant = participant, 
         label = label, 
         category = category)



# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df <- tibble()
gyr_df <- tibble()

acc_set <- 1
gyr_set <- 1

for (f in files) {
  participant <- f %>% 
    str_replace(data_path, "") %>% 
    str_split("-") %>% 
    pluck(1, 1)

  label <- f %>% 
    str_split("-") %>% 
    pluck(1, 2)

  category <- f %>% 
    str_split("-") %>% 
    pluck(1, 3) %>% 
    str_remove("123") %>% 
    str_remove("_MetaWear_2019")

  df <- read_csv(f) %>%
    mutate(participant = participant, 
           label = label, 
           category = category)

  if (str_detect(f, "Accelerometer")) {
    df <- df %>% mutate(set = acc_set)
    acc_set <- acc_set + 1
    acc_df <- bind_rows(acc_df, df)
  }

  if (str_detect(f, "Gyroscope")) {
    df <- df %>% mutate(set = gyr_set)
    gyr_set <- gyr_set + 1
    gyr_df <- bind_rows(gyr_df, df)
  }
}

