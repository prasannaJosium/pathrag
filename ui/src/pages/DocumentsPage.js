import React, { useState, useEffect, useRef } from 'react';
import { Button, Modal, Message, toaster } from 'rsuite';
import Layout from '../components/Layout';
import DocumentList from '../components/documents/DocumentList';
import DocumentUploader from '../components/documents/DocumentUploader';
import { documentAPI } from '../services/api';
import '../styles/documents.css';

const DocumentsPage = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [reloading, setReloading] = useState(false);
  const [error, setError] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const uploaderRef = useRef(null);

  // Fetch documents
  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await documentAPI.getDocuments();
      setDocuments(response.data.documents);
    } catch (error) {
      console.error('Error fetching documents:', error);
      setError('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  // Fetch documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Handle upload complete
  const handleUploadComplete = (document) => {
    setShowUploadModal(false);
    fetchDocuments();
  };

  // Reload PathRAG to recognize new documents
  const handleReloadDocuments = async () => {
    try {
      setReloading(true);
      const response = await documentAPI.reloadDocuments();
      if (response && response.data && response.data.success) {
        toaster.push(
          <Message type="success">
            {response.data.message || 'Documents reloaded successfully'}
          </Message>,
          { placement: 'topCenter', duration: 3000 }
        );
      }
    } catch (error) {
      console.error('Error reloading documents:', error);
      toaster.push(
        <Message type="error">
          Failed to reload documents. Please try again.
        </Message>,
        { placement: 'topCenter', duration: 3000 }
      );
    } finally {
      setReloading(false);
    }
  };

  // Handle delete document
  const handleDeleteDocument = async (id) => {
    if (!window.confirm('Are you sure you want to delete this document? This will remove all related graph data.')) {
      return;
    }

    try {
      await documentAPI.deleteDocument(id);
      setDocuments(documents.filter(doc => doc.id !== id));
      toaster.push(
        <Message type="success">Document deleted successfully</Message>,
        { placement: 'topCenter', duration: 3000 }
      );
    } catch (error) {
      console.error('Error deleting document:', error);
      toaster.push(
        <Message type="error">Failed to delete document</Message>,
        { placement: 'topCenter', duration: 3000 }
      );
    }
  };

  return (
    <Layout>
      <div className="documents-header">
        <h2>Documents</h2>
        <div className="document-actions">
          <Button
            appearance="ghost"
            onClick={handleReloadDocuments}
            disabled={reloading}
            style={{ marginRight: '10px' }}
          >
            {reloading ? 'Reloading...' : 'Reload Documents'}
          </Button>
          <Button appearance="primary" onClick={() => setShowUploadModal(true)}>
            Upload Document
          </Button>
        </div>
      </div>

      {error && <Message type="error">{error}</Message>}

      <DocumentList documents={documents} loading={loading} onDelete={handleDeleteDocument} />

      <Modal
        open={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        size="md"
      >
        <Modal.Header>
          <Modal.Title>Upload Document</Modal.Title>
        </Modal.Header>

        <Modal.Body>
          <DocumentUploader ref={uploaderRef} onUploadComplete={handleUploadComplete} />
        </Modal.Body>

        <Modal.Footer>
          <Button onClick={() => uploaderRef.current?.open()} appearance="primary" style={{ marginRight: '10px' }}>
            Browse Files
          </Button>
          <Button onClick={() => setShowUploadModal(false)} appearance="subtle">
            Cancel
          </Button>
        </Modal.Footer>
      </Modal>
    </Layout>
  );
};

export default DocumentsPage;
