import { useState, useEffect, useCallback, useRef } from "react";
import { Upload, Send, ChevronDown, ChevronUp, ThumbsUp, ThumbsDown } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import { useDebounce } from "use-debounce";

const Index = () => {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [debouncedQuestion] = useDebounce(question, 500); // Debounce queries by 500ms
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [retrievedDocuments, setRetrievedDocuments] = useState<any[]>([]);
  const [relevantDocumentIds, setRelevantDocumentIds] = useState<any[]>([]);
  const [isDocumentsOpen, setIsDocumentsOpen] = useState(false);
  const [isRelevantOpen, setIsRelevantOpen] = useState(false);
  const previousAnswers = useRef<Map<string, any>>(new Map());
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedback, setFeedback] = useState<"positive" | "negative" | null>(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);

  const title = "PDF.STuDiO";
  const API_URL = "http://127.0.0.1:8001";

  // Document preloading hook
  const useDocumentPreload = (documents: string[]) => {
    useEffect(() => {
      documents.forEach((doc, index) => {
        queryClient.setQueryData(['document', index], doc);
      });
    }, [documents, queryClient]);
  };

  // File upload mutation with optimistic updates
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return axios.post(`${API_URL}/process`, formData);
    },
    onMutate: () => {
      setIsProcessing(true);
    },
    onSettled: () => {
      setIsProcessing(false);
    },
    onSuccess: () => {
      toast({
        title: "Document Processed",
        description: "Your PDF has been indexed successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error Processing PDF",
        description: error.message || "An error occurred.",
        variant: "destructive",
      });
    },
  });

  // Query mutation with optimistic updates
  const queryMutation = useMutation({
    mutationFn: async (queryText: string) => {
      const formData = new FormData();
      formData.append("prompt", queryText);
      return axios.post(`${API_URL}/ask`, formData);
    },
    onMutate: async (newQuery) => {
      setIsAsking(true);
      await queryClient.cancelQueries({ queryKey: ['query', debouncedQuestion] });
      const previousQuery = queryClient.getQueryData(['query', debouncedQuestion]);
      queryClient.setQueryData(['query', debouncedQuestion], {
        response: 'Loading...',
        retrieved_documents: [],
        relevant_ids: [],
      });
      return { previousQuery };
    },
    onSuccess: (data: any) => {
      setAnswer(data.data.response);
      setRetrievedDocuments(data.data.retrieved_documents);
      setRelevantDocumentIds(data.data.relevant_ids);
      toast({
        title: "Answer Received",
        description: "Your question has been processed successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error Getting Answer",
        description: error.message || "An error occurred.",
        variant: "destructive",
      });
    },
    onSettled: () => {
      setIsAsking(false);
    },
  });

  // Feedback mutation
  const feedbackMutation = useMutation({
    mutationFn: async (feedbackData: { rating: "positive" | "negative", comment?: string }) => {
      return axios.post(`${API_URL}/feedback`, feedbackData);
    },
    onSuccess: () => {
      toast({
        title: "Thank you for your feedback!",
        description: "Your response helps improve the AI.",
      });
      setShowFeedback(false);
      setFeedback(null);
      setFeedbackComment("");
    },
    onError: (error: any) => {
      toast({
        title: "Error submitting feedback",
        description: error.message || "Failed to submit feedback",
        variant: "destructive",
      });
    },
  });

  /** Handle File Selection */
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      uploadMutation.mutate(selectedFile);
      toast({
        title: "File uploaded successfully",
        description: selectedFile.name,
      });
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF file",
        variant: "destructive",
      });
    }
  };

  /** Ask a Question */
  const handleAsk = async () => {
    if (!debouncedQuestion) return;
    queryMutation.mutate(debouncedQuestion);
  };

  const handleFeedback = async () => {
    if (!feedback) return;
    setIsSubmittingFeedback(true);
    feedbackMutation.mutate({
      rating: feedback,
      comment: feedbackComment.trim() || undefined,
    });
    setIsSubmittingFeedback(false);
  };

  // Preload documents when they're retrieved
  useDocumentPreload(retrievedDocuments);

  return (
    <div className="min-h-screen p-8 animate-fade-in">
      <div className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-white/[0.05] shadow-lg">
        <div className="container max-w-5xl mx-auto py-6">
          <h1 className="text-5xl font-black tracking-[-0.02em] text-white">
            {title.split('').map((char, index) => (
              <span key={index} className="animate-letter inline-block">
                {char}
              </span>
            ))}
          </h1>
        </div>
      </div>

      <main className="container max-w-5xl mx-auto pt-24">
        <div className="grid md:grid-cols-[320px,1fr] gap-8">
          <div className="glass-panel p-8 space-y-8 h-fit">
            <div className="space-y-6">
              <h2 className="text-2xl font-medium text-white/90">
                Upload PDF
              </h2>
              <label className="flex flex-col items-center justify-center w-full h-40 glass-panel cursor-pointer hover:bg-white/[0.04] group">
                <div className="flex flex-col items-center justify-center pt-5 pb-6 space-y-3">
                  <div className="p-4 rounded-full bg-primary/10 group-hover:bg-primary/20 transition-colors">
                    <Upload className="w-6 h-6 text-primary" />
                  </div>
                  <p className="text-sm text-white/70">
                    {file ? file.name : "Drop your PDF here or click to browse"}
                  </p>
                </div>
                <input
                  type="file"
                  className="hidden"
                  accept=".pdf"
                  onChange={handleFileUpload}
                  disabled={isProcessing}
                />
              </label>
              {isProcessing && (
                <div className="text-center text-white/70 animate-pulse">
                  Processing document...
                </div>
              )}
            </div>
          </div>

          <div className="space-y-8">
            <div className="glass-panel p-8 space-y-8">
              <div className="space-y-6">
                <textarea
                  className="glass-input w-full h-40 resize-none"
                  placeholder="Ask a question related to your document..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={isAsking}
                />
                <button
                  className="glass-button w-full flex items-center justify-center gap-3 group"
                  onClick={handleAsk}
                  disabled={!question || isAsking}
                >
                  <Send className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  {isAsking ? "Processing..." : "Ask Question"}
                </button>
              </div>

              {answer && (
                <div className="p-6 glass-panel space-y-4 animate-fade-in">
                  <h3 className="font-medium">Answer:</h3>
                  <p className="text-white/70">{answer}</p>
                  
                  {/* Feedback Section */}
                  {!showFeedback ? (
                    <div className="flex items-center gap-4 mt-4">
                      <button
                        onClick={() => setShowFeedback(true)}
                        className="text-sm text-white/60 hover:text-white/80 transition-colors"
                      >
                        Was this answer helpful?
                      </button>
                    </div>
                  ) : (
                    <div className="mt-4 space-y-4">
                      <div className="flex items-center gap-4">
                        <button
                          onClick={() => setFeedback("positive")}
                          className={`p-2 rounded-full transition-colors ${
                            feedback === "positive"
                              ? "bg-green-500/20 text-green-500"
                              : "hover:bg-white/10"
                          }`}
                        >
                          <ThumbsUp className="w-5 h-5" />
                        </button>
                        <button
                          onClick={() => setFeedback("negative")}
                          className={`p-2 rounded-full transition-colors ${
                            feedback === "negative"
                              ? "bg-red-500/20 text-red-500"
                              : "hover:bg-white/10"
                          }`}
                        >
                          <ThumbsDown className="w-5 h-5" />
                        </button>
                      </div>
                      
                      {feedback && (
                        <div className="space-y-2">
                          <textarea
                            placeholder="Any additional comments? (optional)"
                            value={feedbackComment}
                            onChange={(e) => setFeedbackComment(e.target.value)}
                            className="glass-input w-full h-20 resize-none"
                          />
                          <div className="flex gap-2">
                            <button
                              onClick={handleFeedback}
                              disabled={isSubmittingFeedback}
                              className="glass-button flex-1"
                            >
                              {isSubmittingFeedback ? "Submitting..." : "Submit Feedback"}
                            </button>
                            <button
                              onClick={() => {
                                setShowFeedback(false);
                                setFeedback(null);
                                setFeedbackComment("");
                              }}
                              className="glass-button"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              <button
                className="flex items-center justify-between w-full p-4 glass-panel hover:bg-white/[0.04] transition-colors"
                onClick={() => setIsDocumentsOpen(!isDocumentsOpen)}
              >
                <span className="font-medium">Retrieved Documents</span>
                {isDocumentsOpen ? <ChevronUp /> : <ChevronDown />}
              </button>
              {isDocumentsOpen && retrievedDocuments.length > 0 && (
                <div className="p-6 glass-panel space-y-4 animate-fade-in">
                  {retrievedDocuments.map((doc, index) => (
                    <p key={index} className="text-white/60">{doc}</p>
                  ))}
                </div>
              )}

              <button
                className="flex items-center justify-between w-full p-4 glass-panel hover:bg-white/[0.04] transition-colors"
                onClick={() => setIsRelevantOpen(!isRelevantOpen)}
              >
                <span className="font-medium">Relevant Document IDs</span>
                {isRelevantOpen ? <ChevronUp /> : <ChevronDown />}
              </button>
              {isRelevantOpen && relevantDocumentIds.length > 0 && (
                <div className="p-6 glass-panel space-y-4 animate-fade-in">
                  {relevantDocumentIds.map((id, index) => (
                    <p key={index} className="text-white/60">{id}</p>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
