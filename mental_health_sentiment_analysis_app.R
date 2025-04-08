# ========================
# Load Required Libraries
# ========================
library(tidyverse)
library(tidytext)
library(stringr)
library(readr)
library(textclean)
library(tm)
library(ggplot2)
library(wordcloud)
library(wordcloud2)
library(DT)
library(shiny)
library(reshape2)
library(viridis)
library(caret)
library(e1071)

# ========================
# Load Dataset
# ========================
data <- read_csv("C:/girgit logs/VESIT/SEM 6/R Project/archive/Combined Data.csv")
colnames(data) <- c("id", "statement", "status")

# ========================
# Clean & Preprocess Text
# ========================
data <- data %>%
  mutate(cleaned_text = statement %>%
           tolower() %>%
           replace_contraction() %>%
           replace_symbol() %>%
           removePunctuation() %>%
           removeNumbers() %>%
           stripWhitespace())

# ========================
# Tokenization
# ========================
tokens <- data %>%
  select(id, status, cleaned_text) %>%
  unnest_tokens(word, cleaned_text) %>%
  anti_join(get_stopwords(), by = "word")

# ========================
# Word Frequency
# ========================
word_freq <- tokens %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = 20)

# ========================
# Bing Sentiment Analysis
# ========================
bing_scores <- tokens %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  mutate(score = ifelse(sentiment == "positive", 1, -1)) %>%
  group_by(statement_id = row_number()) %>%
  summarise(sentiment_score = sum(score)) %>%
  ungroup()

data <- data %>%
  mutate(statement_id = row_number()) %>%
  left_join(bing_scores, by = "statement_id") %>%
  mutate(sentiment_score = replace_na(sentiment_score, 0),
         sentiment_label = case_when(
           sentiment_score > 0 ~ "Positive",
           sentiment_score < 0 ~ "Negative",
           TRUE ~ "Neutral"
         ))

# ========================
# NRC Emotion Lexicon
# ========================
nrc_lexicon <- read_tsv("C:/girgit logs/VESIT/SEM 6/R Project/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", col_names = FALSE)
colnames(nrc_lexicon) <- c("word", "emotion", "value")
nrc_filtered <- nrc_lexicon %>% filter(value == 1) %>% select(-value)

emotion_words <- tokens %>%
  inner_join(nrc_filtered, by = "word")

emotion_counts <- emotion_words %>%
  count(emotion, sort = TRUE)

# ========================
# Emotion by Status
# ========================
emotion_by_status <- tokens %>%
  inner_join(nrc_filtered, by = "word") %>%
  count(status, emotion, sort = TRUE)

emotion_summary <- emotion_by_status %>%
  pivot_wider(names_from = status, values_from = n, values_fill = 0)

emotion_melted <- melt(emotion_summary, id.vars = "emotion", variable.name = "status", value.name = "count")

# ========================
# Model Training - Status Prediction
# ========================
set.seed(123)
dtm <- DocumentTermMatrix(Corpus(VectorSource(data$cleaned_text)))
dtm <- removeSparseTerms(dtm, 0.99)
mat <- as.matrix(dtm)
model_data <- as.data.frame(mat)
model_data$status <- as.factor(data$status)

split <- createDataPartition(model_data$status, p = 0.8, list = FALSE)
train_data <- model_data[split, ]
test_data <- model_data[-split, ]

nb_model <- naiveBayes(status ~ ., data = train_data)
predictions <- predict(nb_model, test_data)
model_accuracy <- confusionMatrix(predictions, test_data$status)

predict_status <- function(input_text) {
  clean <- input_text %>%
    tolower() %>%
    replace_contraction() %>%
    replace_symbol() %>%
    removePunctuation() %>%
    removeNumbers() %>%
    stripWhitespace()
  dtm_input <- DocumentTermMatrix(Corpus(VectorSource(clean)), control = list(dictionary = Terms(dtm)))
  input_matrix <- as.data.frame(as.matrix(dtm_input))
  missing_cols <- setdiff(colnames(train_data)[-ncol(train_data)], colnames(input_matrix))
  for (col in missing_cols) input_matrix[[col]] <- 0
  input_matrix <- input_matrix[, colnames(train_data)[-ncol(train_data)]]
  pred <- predict(nb_model, input_matrix)
  return(pred)
}

# ========================
# SHINY DASHBOARD STARTS
# ========================
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { background-color: #121212; color: #ffffff; font-family: 'Arial'; }
      .well { background-color: #1e1e1e; border: none; }
      .selectize-input { background-color: #1e1e1e; color: white; }
      .dataTables_wrapper { color: #ffffff; }
      h2, h3, h4 { color: #00e6e6; }
    "))
  ),
  
  titlePanel("Mental Health Sentiment & Emotion Analysis"),
  
  fluidRow(
    column(12, h3("Analyzing Sentiments to Understand Mental Health Better", style = "color:#00e6e6"))
  ),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("view", "Choose View:",
                  choices = c("Top Words", "Sentiment", "Emotion", "Summary Table", "Predict Status")),
      checkboxGroupInput("status_filter", "Filter by Status:",
                         choices = unique(data$status),
                         selected = unique(data$status)),
      conditionalPanel(
        condition = "input.view == 'Predict Status'",
        textAreaInput("input_text", "Enter a mental health-related statement:", height = "100px"),
        actionButton("predict_btn", "Predict Status", class = "btn-primary")
      )
    ),
    
    mainPanel(
      conditionalPanel(
        condition = "input.view == 'Top Words'",
        fluidRow(
          column(6, plotOutput("topWordsPlot")),
          column(6, wordcloud2Output("wordCloud", height = "400px"))
        )
      ),
      conditionalPanel(condition = "input.view == 'Sentiment'", plotOutput("sentimentPlot")),
      conditionalPanel(condition = "input.view == 'Emotion'", plotOutput("emotionPlot"), plotOutput("emotionHeatmap")),
      conditionalPanel(condition = "input.view == 'Summary Table'", dataTableOutput("summaryTable")),
      conditionalPanel(condition = "input.view == 'Predict Status'", verbatimTextOutput("predictionResult"))
    )
  )
)

server <- function(input, output) {
  
  filtered_tokens <- reactive({
    tokens %>% filter(status %in% input$status_filter)
  })
  
  output$topWordsPlot <- renderPlot({
    filtered_tokens() %>%
      count(word, sort = TRUE) %>%
      slice_max(n, n = 20) %>%
      ggplot(aes(x = reorder(word, n), y = n)) +
      geom_col(fill = "#00e6e6") +
      coord_flip() +
      labs(title = "Top 20 Most Frequent Words", x = "Word", y = "Frequency") +
      theme_minimal() +
      theme(plot.background = element_rect(fill = "#121212", color = NA),
            panel.background = element_rect(fill = "#121212", color = NA),
            text = element_text(color = "#ffffff"))
  })
  
  output$wordCloud <- renderWordcloud2({
    wc_data <- filtered_tokens() %>%
      count(word, sort = TRUE) %>%
      rename(freq = n) %>%
      filter(!is.na(word) & word != "") %>%
      slice_max(freq, n = 100)
    if (nrow(wc_data) == 0) return(NULL)
    wordcloud2(wc_data, size = 1.5, color = "random-light", backgroundColor = "#2b2b2b")
  })
  
  output$sentimentPlot <- renderPlot({
    data %>%
      filter(status %in% input$status_filter) %>%
      count(sentiment_label) %>%
      ggplot(aes(x = sentiment_label, y = n, fill = sentiment_label)) +
      geom_col(show.legend = FALSE) +
      scale_fill_manual(values = c("Positive" = "#00a878", "Negative" = "#f45b69", "Neutral" = "#ffc300")) +
      labs(title = "Overall Sentiment Distribution", x = "Sentiment", y = "Count") +
      theme_minimal() +
      theme(plot.background = element_rect(fill = "#121212", color = NA),
            panel.background = element_rect(fill = "#121212", color = NA),
            text = element_text(color = "#ffffff"))
  })
  
  output$emotionPlot <- renderPlot({
    emotion_melted %>%
      filter(status %in% input$status_filter) %>%
      ggplot(aes(x = status, y = count, fill = emotion)) +
      geom_bar(stat = "identity") +
      labs(title = "Emotion Distribution by Status", x = "Status", y = "Count") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.background = element_rect(fill = "#121212", color = NA),
            panel.background = element_rect(fill = "#121212", color = NA),
            text = element_text(color = "#ffffff"))
  })
  
  output$emotionHeatmap <- renderPlot({
    emotion_melted %>%
      filter(status %in% input$status_filter) %>%
      ggplot(aes(x = status, y = emotion, fill = count)) +
      geom_tile(color = "white") +
      scale_fill_viridis(name = "Emotion Count") +
      labs(title = "Emotion Heatmap by Status", x = "Status", y = "Emotion") +
      theme_minimal() +
      theme(plot.background = element_rect(fill = "#121212", color = NA),
            panel.background = element_rect(fill = "#121212", color = NA),
            text = element_text(color = "#ffffff"))
  })
  
  output$summaryTable <- renderDataTable({
    datatable(emotion_summary,
              options = list(pageLength = 25),
              class = 'display',
              style = "bootstrap")
  })
  
  observeEvent(input$predict_btn, {
    output$predictionResult <- renderText({
      req(input$input_text)
      pred <- predict_status(input$input_text)
      paste("Predicted Mental Health Status:", pred)
    })
  })
}

# ========================
# Run the App
# ========================
shinyApp(ui = ui, server = server)
