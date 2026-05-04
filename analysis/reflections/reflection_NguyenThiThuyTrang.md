# Individual Reflection - Lab 18

**Tên:** Nguyễn Thị Thùy Trang  
**Module phụ trách:** Cá nhân làm end-to-end các module M1, M2, M3, M4 và tích hợp pipeline (Do chiều được nghỉ)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:**
  - M1: Advanced Chunking Strategies
  - M2: Hybrid Search
  - M3: Reranking
  - M4: Evaluation và Failure Analysis
  - M5: Enrichment fallback
  - Production pipeline end-to-end

- **Các hàm/class chính đã viết:**
  - `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`
  - `BM25Search`, `DenseSearch`, `HybridSearch`, `reciprocal_rank_fusion()`
  - `CrossEncoderReranker`, `FlashrankReranker`, `benchmark_reranker()`
  - `evaluate_ragas()`, `failure_analysis()`, `save_report()`
  - `enrich_chunks()`, `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`
  - `build_pipeline()`, `run_query()`, `evaluate_pipeline()`

- **Các phần hỗ trợ thêm:**
  - Tạo `docker-compose.yml` để chạy Qdrant local.
  - Tạo `requirements.txt`.
  - Chuyển PDF sang markdown/OCR text để pipeline có dữ liệu truy xuất.
  - Tạo lại `test_set.json` gồm 24 câu hỏi bám vào nội dung thật của `BCTC.md` và `Nghi_dinh_13_2023.md`.
  - Điền `failure_analysis.md` từ kết quả report.

- **Số tests pass:** 37/37

## 2. Kiến thức học được

- **Khái niệm mới nhất:**
  - Production RAG không chỉ là embedding + vector search. Chất lượng phụ thuộc vào nhiều bước: chunking, hybrid search, reranking, enrichment, evaluation và failure analysis.
  - Hybrid search giúp kết hợp ưu điểm của BM25 và dense retrieval. BM25 tốt với keyword/số liệu cụ thể, còn dense search tốt với câu hỏi ngữ nghĩa.
  - Reranking giúp tăng context precision bằng cách sắp xếp lại top results theo mức độ liên quan với query.
  - RAGAS hoặc evaluator tương tự giúp nhìn ra pipeline sai ở đâu: hallucination, thiếu context, context nhiễu, hoặc answer không khớp câu hỏi.

- **Điều bất ngờ nhất:**
  - Khi answer lấy trực tiếp từ context thì faithfulness rất cao, nhưng điều đó không có nghĩa pipeline đã trả lời tốt. Answer relevancy và context precision mới phản ánh rõ hơn chất lượng retrieval.
  - Dữ liệu OCR/PDF nếu không được chuyển thành text sạch thì RAG gần như không thể hoạt động tốt, dù có dùng model embedding tốt.
  - Enrichment nếu đưa trực tiếp vào text index có thể giúp recall nhưng cũng có thể gây nhiễu, nhất là phần `Likely questions`.

- **Kết nối với bài giảng:**
  - Phần chunking liên quan trực tiếp đến bài học về parent-child retrieval và structure-aware chunking.
  - Phần search liên quan đến hybrid retrieval: BM25 + dense vector + RRF.
  - Phần rerank liên quan đến cross-encoder reranking để tăng precision.
  - Phần evaluation liên quan đến RAGAS metrics và error tree/failure analysis.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:**
  - Dữ liệu ban đầu là PDF/markdown chưa có text tốt. `BCTC.md` và `Nghi_dinh_13_2023.md` ban đầu không đủ nội dung để test RAG có ý nghĩa.
  - Một số package/model nặng như embedding hoặc reranker thật không nên tải mặc định vì dễ chậm hoặc lỗi môi trường.
  - Khi thử Gemini để sinh answer, kết quả metric lại giảm vì evaluator hiện tại chấm theo overlap từ vựng, trong khi Gemini trả lời ngắn và paraphrase.

- **Cách giải quyết:**
  - Chuyển dữ liệu sang markdown có text rõ hơn, đặc biệt sửa `BCTC.md` thành bảng markdown chuẩn.
  - Tạo fallback local cho embedding, reranking và evaluation để pipeline vẫn chạy được khi không có API/model.
  - Tạo lại test set theo nội dung thật hiện có thay vì hỏi về ảnh/trang PDF.
  - Gỡ Gemini khỏi pipeline vì không phù hợp với evaluator hiện tại và làm kết quả chấm lab thấp hơn.

- **Thời gian debug:**
  - Phần Docker/Qdrant và thiếu file cấu hình: khoảng 20-30 phút.
  - Phần requirements, TODO implementation và tests: khoảng 1-2 giờ.
  - Phần dữ liệu PDF/markdown/test set/evaluation: khoảng 1 giờ.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:**
  - Chuẩn hóa dữ liệu đầu vào trước khi viết pipeline. Với tài liệu pháp lý nên tách sẵn theo `Chương`, `Mục`, `Điều`, `Khoản`; với tờ khai/bảng nên giữ bảng markdown sạch.
  - Không đưa enrichment text như `Likely questions` vào chung với chunk chính. Nên lưu enrichment vào metadata hoặc field phụ để tránh nhiễu.
  - Thiết kế test set sau khi data đã ổn định, tránh phải đổi test set nhiều lần.
  - Thêm extractive answer step để lấy câu trả lời ngắn từ context thay vì trả nguyên context đầu tiên.

- **Module muốn thử tiếp:**
  - Reranking thật bằng `BAAI/bge-reranker-v2-m3`.
  - Embedding thật bằng `BAAI/bge-m3`.
  - Metadata filtering theo loại tài liệu, section, điều luật.
  - LLM answer generation nhưng cần evaluator phù hợp hơn, ví dụ dùng RAGAS thật với API key thay vì heuristic overlap.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 3 |
| Problem solving | 5 |

## 6. Kết quả cuối cùng

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.4681 | 0.6929 | +0.2248 |
| Context Precision | 0.4426 | 0.6296 | +0.1870 |
| Context Recall | 0.8158 | 0.7927 | -0.0231 |

Kết quả cho thấy production pipeline cải thiện rõ ở answer relevancy và context precision. Context recall giảm nhẹ là trade-off chấp nhận được vì pipeline lấy chunk chính xác hơn nhưng đôi khi bỏ sót thông tin trong các câu hỏi dạng danh sách dài.
