import React, { useCallback, useRef, useState } from "react";

interface ImageUploadProps {
  images: File[];
  onChange: (images: File[]) => void;
  compact?: boolean;
  trigger?: boolean;  // If true, renders just a trigger button
}

const MAX_IMAGES = 5;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/tiff", "application/pdf"];

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

function ThumbnailPreview({ file, onRemove }: { file: File; onRemove: () => void }) {
  const [url, setUrl] = useState<string | null>(null);

  React.useEffect(() => {
    if (file.type.startsWith("image/")) {
      const u = URL.createObjectURL(file);
      setUrl(u);
      return () => URL.revokeObjectURL(u);
    }
  }, [file]);

  return (
    <div className="relative group inline-block">
      <div className="w-16 h-16 rounded-lg border border-gray-700 bg-gray-800 overflow-hidden flex items-center justify-center">
        {url ? (
          <img src={url} alt={file.name} className="w-full h-full object-cover" />
        ) : (
          <div className="flex flex-col items-center justify-center p-1">
            <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="text-xs text-gray-500 truncate w-full text-center px-1">{file.name.slice(0, 6)}</span>
          </div>
        )}
      </div>
      <button
        onClick={onRemove}
        className="absolute -top-1 -right-1 w-4 h-4 bg-red-600 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
      <p className="text-xs text-gray-600 text-center mt-0.5 truncate w-16">
        {formatSize(file.size)}
      </p>
    </div>
  );
}

export default function ImageUpload({ images, onChange, compact, trigger }: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const addFiles = useCallback((files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) => ACCEPTED_TYPES.includes(f.type));
    const remaining = MAX_IMAGES - images.length;
    if (remaining <= 0) return;
    onChange([...images, ...arr.slice(0, remaining)]);
  }, [images, onChange]);

  const removeImage = (idx: number) => {
    onChange(images.filter((_, i) => i !== idx));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) addFiles(e.dataTransfer.files);
  };

  // Trigger button only (icon button for input area)
  if (trigger) {
    return (
      <>
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          multiple
          className="hidden"
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={images.length >= MAX_IMAGES}
          className="px-3 py-3 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 rounded-xl text-gray-300 transition-colors"
          title="Upload documents"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
          </svg>
        </button>
      </>
    );
  }

  // Compact: just show thumbnails
  if (compact) {
    return (
      <div className="flex gap-2 flex-wrap">
        {images.map((file, i) => (
          <ThumbnailPreview key={i} file={file} onRemove={() => removeImage(i)} />
        ))}
      </div>
    );
  }

  // Full upload zone
  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        multiple
        className="hidden"
        onChange={(e) => e.target.files && addFiles(e.target.files)}
      />

      {/* Drop zone */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-6 cursor-pointer transition-all text-center ${
          isDragging
            ? "border-blue-500 bg-blue-950"
            : "border-gray-700 hover:border-gray-600 bg-gray-800"
        }`}
      >
        <svg className="w-8 h-8 text-gray-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <p className="text-sm text-gray-400">Drag & drop or click to upload</p>
        <p className="text-xs text-gray-600 mt-1">
          JPG, PNG, PDF · Max {MAX_IMAGES} files
        </p>
        {images.length >= MAX_IMAGES && (
          <p className="text-xs text-orange-400 mt-1">Maximum {MAX_IMAGES} files reached</p>
        )}
      </div>

      {/* Previews */}
      {images.length > 0 && (
        <div className="mt-3 flex gap-3 flex-wrap">
          {images.map((file, i) => (
            <ThumbnailPreview key={i} file={file} onRemove={() => removeImage(i)} />
          ))}
        </div>
      )}
    </div>
  );
}
