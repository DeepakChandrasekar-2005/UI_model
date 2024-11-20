export const openDatabase = async (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("FaceAuthDB", 1);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains("uploads")) {
        db.createObjectStore("uploads", { keyPath: "id", autoIncrement: true });
      }
    };

    request.onsuccess = (event) => {
      resolve((event.target as IDBOpenDBRequest).result);
    };

    request.onerror = (event) => {
      reject(
        `Error opening IndexedDB: ${
          (event.target as IDBOpenDBRequest).error?.message
        }`
      );
    };
  });
};

export const saveImageToIndexedDB = async (
  base64Image: string,
  folder: string
): Promise<void> => {
  try {
    const db = await openDatabase();
    const transaction = db.transaction("uploads", "readwrite");
    const store = transaction.objectStore("uploads");

    const imageData = {
      folder,
      image: base64Image,
      timestamp: new Date().toISOString(),
    };

    store.add(imageData);

    return new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => {
        console.log("Transaction completed. Image saved successfully!");
        resolve();
      };

      transaction.onerror = (event) => {
        console.error(
          "Transaction error:",
          (event.target as IDBRequest).error?.message
        );
        reject(
          `Error saving image: ${(event.target as IDBRequest).error?.message}`
        );
      };
    });
  } catch (error) {
    console.error("Error saving image to IndexedDB:", error);
    throw error;
  }
};

export const getImageFromIndexedDB = async (
  folder: string
): Promise<string | null> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("FaceAuthDB");

    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(["uploads"], "readonly");
      const store = transaction.objectStore("uploads");
      const getRequest = store.get(folder);

      getRequest.onsuccess = () => {
        const result = getRequest.result;
        if (result && result.image) {
          resolve(result.image); // Returns the base64 image
        } else {
          resolve(null); // No image found
        }
      };

      getRequest.onerror = () => {
        reject("Failed to retrieve image from IndexedDB.");
      };
    };

    request.onerror = () => {
      reject("Failed to open IndexedDB.");
    };
  });
};
