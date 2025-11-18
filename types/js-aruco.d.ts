declare module 'js-aruco' {
  export namespace AR {
    type Corner = { x: number; y: number };
    type Marker = { id?: number; corners: Corner[] };

    class Detector {
      detect(imageData: ImageData): Marker[];
    }
  }

  export const AR: {
    Detector: new () => {
      detect(imageData: ImageData): AR.Marker[];
    };
  };
}
