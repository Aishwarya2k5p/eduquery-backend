import os
import logging
from fastapi import UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from EduQuery.pdf_processor import PDFProcessor
from EduQuery.chat_manager import ChatManager
from EduQuery.query_processor import QueryProcessor
from EduQuery.utils import is_casual_query, handle_casual_query, rewrite_followup_query, is_followup_query
from EduQuery.config import Config

def setup_routes(app, collection):
    chat_manager = ChatManager()
    pdf_processor = PDFProcessor(collection)
    query_processor = QueryProcessor(collection, chat_manager)

    @app.get("/")
    async def root():
        return {"message": "EduQuery API is running", "status": "ok"}

    @app.post("/upload/")
    async def upload_pdf(file: UploadFile = File(...)):
        try:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            file_path = os.path.join(Config.PDF_DIR, file.filename)
            if os.path.exists(file_path):
                raise HTTPException(status_code=400, detail="A file with this name already exists")
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            pdf_processor.process_pdf(file_path)
            return JSONResponse(status_code=200, content={"message": "PDF uploaded and processed successfully", "filename": file.filename})
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error uploading PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {str(e)}")

    @app.get("/books/")
    async def list_books():
        try:
            books = [b for b in os.listdir(Config.PDF_DIR) if b.endswith('.pdf')]
            return {"books": books}
        except Exception as e:
            logging.error(f"Error listing books: {e}")
            return {"books": [], "error": str(e)}

    @app.get("/book")
    async def get_pdf(subject: str = Query(...)):
        file_path = os.path.join(Config.PDF_DIR, subject)
        if os.path.isfile(file_path):
            return FileResponse(file_path, media_type='application/pdf', filename=subject, headers={"Content-Disposition": "inline"})
        return {"error": "PDF not found"}

    @app.post("/query/")
    async def query_pdf(query: str = Form(...), book_name: str = Form(...), session_id: str = Form(default="default"), debug: bool = Form(default=False)):
        try:
            book_specific_session = f"{session_id}_{book_name}"
            if is_casual_query(query):
                casual_response = handle_casual_query(query)
                chat_manager.add_to_history(book_specific_session, query, casual_response)
                return {"answer": casual_response, "sources": []}
            chat_history = chat_manager.get_history(book_specific_session)
            query_to_use = rewrite_followup_query(query, chat_history)
            if is_followup_query(query) and chat_history:
                last_question = ""
                for entry in reversed(chat_history):
                    q = entry.get("question", "")
                    if q and not is_casual_query(q):
                        last_question = q
                        break
                if last_question:
                    enhanced_query = f"{last_question} {query_to_use}"
                    logging.info(f"Enhanced follow-up query: '{enhanced_query}' (original: '{query}')")
                    query_to_use = enhanced_query
            documents, metadatas = query_processor.retrieve_relevant_content(query_to_use, book_name)
            if not documents:
                return {"answer": "No relevant content found in the PDF for your query.", "sources": []}
            retrieved_texts = "\n\n".join(documents)
            response, query_embedding = query_processor.generate_response(query_to_use, retrieved_texts, book_specific_session, book_name)
            chat_manager.add_to_history(book_specific_session, query, response, query_embedding)
            result = {"answer": response, "sources": [meta.get("source", book_name) for meta in metadatas]}
            if debug:
                result["debug"] = {
                    "chunks_retrieved": len(documents),
                    "total_chunks_in_book": collection.count(),
                    "retrieved_content_preview": retrieved_texts[:1000] + "..." if len(retrieved_texts) > 1000 else retrieved_texts,
                    "book_specific_session": book_specific_session,
                    "cache_enabled": True,
                    "batch_processing": True
                }
            return result
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {"answer": f"Error processing query: {str(e)}", "sources": []}

    @app.post("/end_session/")
    async def end_session(session_id: str = Form(...), book_name: str = Form(...)):
        try:
            file_path = os.path.join(Config.PDF_DIR, book_name)
            removed = False
            if os.path.exists(file_path):
                os.remove(file_path)
                removed = True
            try:
                collection.delete(where={"source": book_name})
                logging.info(f"Successfully removed embeddings for {book_name}")
            except Exception as e:
                logging.error(f"Failed to remove embeddings for {book_name}: {e}")
            if session_id in chat_manager.chat_histories:
                del chat_manager.chat_histories[session_id]
            return {"removed": removed, "filename": book_name, "message": "Session ended successfully"}
        except Exception as e:
            logging.error(f"Error ending session: {e}")
            return {"error": f"Failed to end session: {str(e)}"} 