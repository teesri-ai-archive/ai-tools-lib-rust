#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum PromptPhase {
    System,
    User,
}

impl PromptPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            PromptPhase::System => "system",
            PromptPhase::User => "user",
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum PromptCategory {
    ConversationalAi,
    VideoAi,
    SelectionUtils,
}

impl PromptCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            PromptCategory::ConversationalAi => "conversational-ai",
            PromptCategory::VideoAi => "video-ai",
            PromptCategory::SelectionUtils => "selection-utils",
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum PromptTemplate {
    PromptPreamble,
    UserMessageProcessingSystemPrompt,
    PlanGenerationAgentSystemPrompt,
    DetailedUsageEvaluationAgentSystemPrompt,
    NodeAnalysisPrompt,
    GetEditedVideoResponsePrompt,
    GetEditedVideoResponseUserPrompt,
    ProcessFileMessagePrepPrompt,
    ContentSectioningSystemPrompt,
    DiagramSpeechMatchingPrompt,
    SelectBestImageFromWebpagePrompt,
    GenerateSpeakerIntroBlurbPrompt,
    AnalyzeTranscriptStructurePrompt,
    PickBestSlidePrompt,
    IntroTranscriptPrompt,
    GenerateAddedOutroTranscriptPrompt,
    GenerateSpeakerThankYouBlurbPrompt,
    GenerateNarrativeAdditionForSlidePrompt,
    InsertionHintParserPrompt,
    InsertionHintUserPrompt,
    InsertionHintRetryPrompt,
    ExtractImageDescriptionPrompt,
    ExtractWebpageContentOnlyPrompt,
    ExtractWebpageFullPrompt,
    DslAiHelperSystemPrompt,
    DslAiHelperOtherSourcesSubPrompt,
    SummarizerSystemPrompt,
    ContentEditorPrompt,
    BuildStyleExtractionPrompt,
    ExtractSlideFromGeminiPrompt,
    PronunciationFinderPrompt,
    BestPronunciationSelectionPrompt,
    LocateDisfluenciesPrompt,
    VideoAnalysisPrompt,
    SelectionUtilsSystemPrompt,
    SelectionUtilsUserPrompt,
}

impl PromptTemplate {
    pub fn name(&self) -> &'static str {
        match self {
            PromptTemplate::PromptPreamble => "prompt_preamble",
            PromptTemplate::UserMessageProcessingSystemPrompt => {
                "user_message_processing_system_prompt"
            }
            PromptTemplate::PlanGenerationAgentSystemPrompt => {
                "plan_generation_agent_system_prompt"
            }
            PromptTemplate::DetailedUsageEvaluationAgentSystemPrompt => {
                "detailed_usage_evaluation_agent_system_prompt"
            }
            PromptTemplate::NodeAnalysisPrompt => "node_analysis_prompt",
            PromptTemplate::GetEditedVideoResponsePrompt => "get_edited_video_response_prompt",
            PromptTemplate::GetEditedVideoResponseUserPrompt => {
                "get_edited_video_response_user_prompt"
            }
            PromptTemplate::ProcessFileMessagePrepPrompt => "process_file_message_prep_prompt",
            PromptTemplate::ContentSectioningSystemPrompt => "content_sectioning_system_prompt",
            PromptTemplate::DiagramSpeechMatchingPrompt => "diagram_speech_matching_prompt",
            PromptTemplate::SelectBestImageFromWebpagePrompt => {
                "select_best_image_from_webpage_prompt"
            }
            PromptTemplate::GenerateSpeakerIntroBlurbPrompt => {
                "generate_speaker_intro_blurb_prompt"
            }
            PromptTemplate::AnalyzeTranscriptStructurePrompt => {
                "analyze_transcript_structure_prompt"
            }
            PromptTemplate::PickBestSlidePrompt => "pick_best_slide_prompt",
            PromptTemplate::IntroTranscriptPrompt => "intro_transcript_prompt",
            PromptTemplate::GenerateAddedOutroTranscriptPrompt => {
                "generate_added_outro_transcript_prompt"
            }
            PromptTemplate::GenerateSpeakerThankYouBlurbPrompt => {
                "generate_speaker_thank_you_blurb_prompt"
            }
            PromptTemplate::GenerateNarrativeAdditionForSlidePrompt => {
                "generate_narrative_addition_for_slide_prompt"
            }
            PromptTemplate::InsertionHintParserPrompt => "insertion_hint_parser_prompt",
            PromptTemplate::InsertionHintUserPrompt => "insertion_hint_user_prompt",
            PromptTemplate::InsertionHintRetryPrompt => "insertion_hint_retry_prompt",
            PromptTemplate::ExtractImageDescriptionPrompt => "extract_image_description_prompt",
            PromptTemplate::ExtractWebpageContentOnlyPrompt => {
                "extract_webpage_content_only_prompt"
            }
            PromptTemplate::ExtractWebpageFullPrompt => "extract_webpage_full_prompt",
            PromptTemplate::DslAiHelperSystemPrompt => "dsl_ai_helper_system_prompt",
            PromptTemplate::DslAiHelperOtherSourcesSubPrompt => {
                "dsl_ai_helper_other_sources_sub_prompt"
            }
            PromptTemplate::SummarizerSystemPrompt => "summarizer_system_prompt",
            PromptTemplate::ContentEditorPrompt => "content_editor_prompt",
            PromptTemplate::BuildStyleExtractionPrompt => "build_style_extraction_prompt",
            PromptTemplate::ExtractSlideFromGeminiPrompt => "extract_slide_from_gemini_prompt",
            PromptTemplate::PronunciationFinderPrompt => "pronunciation_finder_prompt",
            PromptTemplate::BestPronunciationSelectionPrompt => {
                "best_pronunciation_selection_prompt"
            }
            PromptTemplate::LocateDisfluenciesPrompt => "locate_disfluencies_prompt",
            PromptTemplate::VideoAnalysisPrompt => "video_analysis_prompt",
            PromptTemplate::SelectionUtilsSystemPrompt => "selection_utils_system_prompt",
            PromptTemplate::SelectionUtilsUserPrompt => "selection_utils_user_prompt",
        }
    }

    pub fn category(&self) -> PromptCategory {
        match self {
            PromptTemplate::PromptPreamble => PromptCategory::ConversationalAi,
            PromptTemplate::UserMessageProcessingSystemPrompt => PromptCategory::ConversationalAi,
            PromptTemplate::PlanGenerationAgentSystemPrompt => PromptCategory::ConversationalAi,
            PromptTemplate::DetailedUsageEvaluationAgentSystemPrompt => {
                PromptCategory::ConversationalAi
            }
            PromptTemplate::NodeAnalysisPrompt => PromptCategory::ConversationalAi,
            PromptTemplate::GetEditedVideoResponsePrompt => PromptCategory::ConversationalAi,
            PromptTemplate::GetEditedVideoResponseUserPrompt => PromptCategory::ConversationalAi,
            PromptTemplate::ProcessFileMessagePrepPrompt => PromptCategory::ConversationalAi,
            PromptTemplate::ContentSectioningSystemPrompt => PromptCategory::VideoAi,
            PromptTemplate::DiagramSpeechMatchingPrompt => PromptCategory::VideoAi,
            PromptTemplate::SelectBestImageFromWebpagePrompt => PromptCategory::VideoAi,
            PromptTemplate::GenerateSpeakerIntroBlurbPrompt => PromptCategory::VideoAi,
            PromptTemplate::AnalyzeTranscriptStructurePrompt => PromptCategory::VideoAi,
            PromptTemplate::PickBestSlidePrompt => PromptCategory::VideoAi,
            PromptTemplate::IntroTranscriptPrompt => PromptCategory::VideoAi,
            PromptTemplate::GenerateAddedOutroTranscriptPrompt => PromptCategory::VideoAi,
            PromptTemplate::GenerateSpeakerThankYouBlurbPrompt => PromptCategory::VideoAi,
            PromptTemplate::GenerateNarrativeAdditionForSlidePrompt => PromptCategory::VideoAi,
            PromptTemplate::InsertionHintParserPrompt => PromptCategory::VideoAi,
            PromptTemplate::InsertionHintUserPrompt => PromptCategory::VideoAi,
            PromptTemplate::InsertionHintRetryPrompt => PromptCategory::VideoAi,
            PromptTemplate::ExtractImageDescriptionPrompt => PromptCategory::VideoAi,
            PromptTemplate::ExtractWebpageContentOnlyPrompt => PromptCategory::VideoAi,
            PromptTemplate::ExtractWebpageFullPrompt => PromptCategory::VideoAi,
            PromptTemplate::DslAiHelperSystemPrompt => PromptCategory::VideoAi,
            PromptTemplate::DslAiHelperOtherSourcesSubPrompt => PromptCategory::VideoAi,
            PromptTemplate::SummarizerSystemPrompt => PromptCategory::VideoAi,
            PromptTemplate::ContentEditorPrompt => PromptCategory::VideoAi,
            PromptTemplate::BuildStyleExtractionPrompt => PromptCategory::VideoAi,
            PromptTemplate::ExtractSlideFromGeminiPrompt => PromptCategory::VideoAi,
            PromptTemplate::PronunciationFinderPrompt => PromptCategory::VideoAi,
            PromptTemplate::BestPronunciationSelectionPrompt => PromptCategory::VideoAi,
            PromptTemplate::LocateDisfluenciesPrompt => PromptCategory::VideoAi,
            PromptTemplate::VideoAnalysisPrompt => PromptCategory::VideoAi,
            PromptTemplate::SelectionUtilsSystemPrompt => PromptCategory::SelectionUtils,
            PromptTemplate::SelectionUtilsUserPrompt => PromptCategory::SelectionUtils,
        }
    }

    pub fn folder_path(&self) -> &'static str {
        self.category().as_str()
    }

    pub fn phase(&self) -> PromptPhase {
        match self {
            PromptTemplate::GetEditedVideoResponseUserPrompt => PromptPhase::User,
            PromptTemplate::InsertionHintUserPrompt => PromptPhase::User,
            PromptTemplate::SelectionUtilsUserPrompt => PromptPhase::User,
            _ => PromptPhase::System,
        }
    }

    pub fn template_file_name(&self) -> &'static str {
        match self.phase() {
            PromptPhase::System => "system_prompt.j2",
            PromptPhase::User => "user_prompt.j2",
        }
    }
}
