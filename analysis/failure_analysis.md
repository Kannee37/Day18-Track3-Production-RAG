# Failure Analysis - Lab 18: Production RAG

**Nhóm:** ...  
**Thành viên:** Nguyễn Thị Thùy Trang

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.4681 | 0.6929 | +0.2248 |
| Context Precision | 0.4426 | 0.6296 | +0.1870 |
| Context Recall | 0.8158 | 0.7927 | -0.0231 |

## Nhận xét tổng quan

Production pipeline cải thiện rõ ở **answer relevancy** và **context precision**. Điều này cho thấy hierarchical chunking, hybrid search và reranking giúp lấy đúng vùng thông tin hơn so với baseline dense-only.

Điểm **context recall** giảm nhẹ vì production ưu tiên các chunk nhỏ và chính xác hơn. Khi câu hỏi cần một danh sách dài hoặc nhiều ý nằm rải rác, pipeline có thể lấy thiếu một phần thông tin.

**Faithfulness** giữ ở mức 1.0 vì answer hiện vẫn được lấy trực tiếp từ retrieved context, gần như không có bước sinh câu trả lời mới nên ít nguy cơ hallucination.

## Bottom-5 Failures

### #1

- **Question:** Theo Nghị định 13/2023, quyền của chủ thể dữ liệu gồm những quyền nào?
- **Expected:** Quyền của chủ thể dữ liệu gồm quyền được biết, quyền đồng ý, quyền truy cập, quyền rút lại sự đồng ý, quyền xóa dữ liệu, quyền hạn chế xử lý dữ liệu, quyền cung cấp dữ liệu, quyền phản đối xử lý dữ liệu và quyền tự bảo vệ.
- **Got:** Pipeline lấy được một phần nội dung về quyền của chủ thể dữ liệu, nhưng context kèm thêm các đoạn không trực tiếp phục vụ câu hỏi.
- **Worst metric:** Context Recall = 0.4483
- **Error Tree:** Output chưa đủ → Context thiếu một phần ý cần trả lời → Query OK → lỗi ở retrieval/chunk selection.
- **Root cause:** Câu hỏi yêu cầu một danh sách nhiều quyền. Các quyền nằm trong cùng một section nhưng chunking/rerank có thể cắt hoặc ưu tiên đoạn phụ, làm context recall thấp.
- **Suggested fix:** Tăng `RERANK_TOP_K` cho các câu hỏi dạng danh sách, giữ nguyên section `Điều 9` thành một chunk đầy đủ, hoặc thêm metadata `section=Điều 9`.

### #2

- **Question:** Tên người nộp thuế trong tờ khai GTGT là gì?
- **Expected:** Tên người nộp thuế là CÔNG TY CỔ PHẦN DHA SURFACES.
- **Got:** Pipeline retrieve nhầm ưu tiên sang phần cam kết/chữ ký trước, dù context phụ có chứa đoạn thông tin tờ khai.
- **Worst metric:** Context Recall = 0.5455
- **Error Tree:** Output chưa trực tiếp → Context đúng một phần → Query OK → lỗi ở ranking.
- **Root cause:** Cụm "người nộp thuế" xuất hiện ở cả phần thông tin công ty và phần chữ ký, nên search/rerank bị nhiễu.
- **Suggested fix:** Thêm metadata theo section, ưu tiên section `Thông tin tờ khai` khi query chứa "tên người nộp thuế", hoặc giảm enrichment text gây nhiễu.

### #3

- **Question:** Ai là người ký tờ khai GTGT?
- **Expected:** Người ký tờ khai GTGT là TRỊNH THỊ SANG.
- **Got:** Pipeline lấy được đoạn chữ ký nhưng context vẫn kèm nhiều nội dung thừa.
- **Worst metric:** Context Precision = 0.4762
- **Error Tree:** Output đúng một phần → Context có đáp án nhưng loãng → Query OK → lỗi ở precision.
- **Root cause:** Chunk chữ ký chứa cả phần cam kết, đại lý thuế, người nộp thuế và ghi chú ký điện tử. Với câu hỏi ngắn, nhiều token thừa làm precision thấp.
- **Suggested fix:** Chunk nhỏ hơn ở phần cuối tài liệu hoặc tách riêng các field như `Người ký`, `Ngày lập`, `Ghi chú`.

### #4

- **Question:** Dữ liệu cá nhân gồm những loại nào?
- **Expected:** Dữ liệu cá nhân bao gồm dữ liệu cá nhân cơ bản và dữ liệu cá nhân nhạy cảm.
- **Got:** Pipeline retrieve được nội dung liên quan đến định nghĩa dữ liệu cá nhân nhưng chưa tập trung đúng câu chứa hai loại dữ liệu.
- **Worst metric:** Context Recall = 0.5455
- **Error Tree:** Output thiếu ý chính → Context gần đúng nhưng chưa đủ → Query OK → lỗi ở retrieval recall.
- **Root cause:** Section `Điều 2` dài, chứa nhiều định nghĩa liền nhau. Chunk nhỏ có thể lấy phần định nghĩa chung nhưng bỏ qua câu phân loại.
- **Suggested fix:** Với văn bản pháp lý, chunk theo điều/khoản thay vì sliding window ký tự; giữ nguyên khoản 1 Điều 2 trong cùng một chunk.

### #5

- **Question:** Nghị định 13/2023/NĐ-CP quy định về vấn đề gì?
- **Expected:** Nghị định quy định về bảo vệ dữ liệu cá nhân và trách nhiệm bảo vệ dữ liệu cá nhân của cơ quan, tổ chức, cá nhân có liên quan.
- **Got:** Pipeline lấy context liên quan đến Nghị định 13 nhưng answer chưa khớp trực tiếp câu hỏi.
- **Worst metric:** Answer Relevancy = 0.5455
- **Error Tree:** Output chưa đúng trọng tâm → Context đúng một phần → Query OK → lỗi ở answer extraction.
- **Root cause:** Pipeline hiện trả về nguyên context đầu tiên thay vì trích ra câu trả lời ngắn. Context có thông tin đúng nhưng answer quá dài và chứa nhiễu.
- **Suggested fix:** Thêm bước extractive answer: sau retrieval, chọn câu có overlap cao nhất với query hoặc dùng prompt LLM có kiểm soát.

## Case Study

**Question chọn phân tích:** Theo Nghị định 13/2023, quyền của chủ thể dữ liệu gồm những quyền nào?

**Error Tree walkthrough:**

1. **Output đúng?** Chưa đầy đủ. Câu hỏi yêu cầu liệt kê nhiều quyền, nhưng answer/context không bao phủ đủ rõ toàn bộ danh sách.
2. **Context đúng?** Đúng một phần. Pipeline có lấy vùng liên quan tới quyền chủ thể dữ liệu, nhưng context recall thấp do thiếu hoặc loãng các quyền cần nêu.
3. **Query rewrite OK?** Query rõ ràng, không cần rewrite. Vấn đề chính nằm ở retrieval/chunking.
4. **Fix ở bước:** Retrieval và chunking. Nên chunk văn bản pháp lý theo `Điều` và `Khoản`, đồng thời tăng ưu tiên metadata section khi query nhắc tới điều luật hoặc chủ đề cụ thể.

## Nếu có thêm 1 giờ, sẽ optimize

- Tách văn bản pháp lý theo cấu trúc `Chương`, `Mục`, `Điều`, `Khoản` thay vì chia theo độ dài ký tự.
- Tạo metadata `document`, `section`, `article_number`, `topic` để filter trước khi search.
- Giảm nhiễu từ enrichment bằng cách không đưa `Likely questions` vào text chính, hoặc lưu vào field riêng.
- Thêm extractive answer step: chọn câu hoặc đoạn ngắn nhất chứa đáp án từ top contexts.
- Tune `RERANK_TOP_K` và `HYBRID_TOP_K` riêng cho câu hỏi dạng định nghĩa/danh sách.
