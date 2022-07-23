using Keras.Layers;
using Keras.Models;

using Python.Runtime;

namespace KerasNET_Generative_Adversarial_Network.src
{
    public static class ExtensionMethods
    {
        /* List of Helper Functions provided by kroll-software AKA: Detlef
         * They can be found at: https://github.com/SciSharp/Keras.NET/issues/141
         * All credits to Detlef! I've spent hours trying to figure this out
         * to build a GAN but could not figure it out on my own.
         */

        public static BaseLayer Call(this BaseModel model, BaseLayer input)
        {
            return new BaseLayer(model.ToPython().InvokeMethod("__call__", input.ToPython()));
        }
        public static void SetTrainable(this BaseModel model, bool trainable)
        {
            model.ToPython().SetAttr("trainable", new PyInt(trainable ? 1 : 0));
        }
        public static bool IsTrainable(this BaseModel model)
        {
            return model.ToPython().GetAttr("trainable").ToString() == "True";
        }
        public static BaseLayer[] Layers(this BaseModel model)
        {
            List<BaseLayer> lstLayers = new List<BaseLayer>();
            var layers = model.ToPython().GetAttr("layers");
            foreach (var layer in layers)
            {
                lstLayers.Add(new BaseLayer(layer as PyObject));
            }
            return lstLayers.ToArray();
        }
        public static void SetLayersTrainable(this BaseModel model, bool trainable, int count = -1)
        {
            PyObject bval = new PyInt(trainable ? 1 : 0);
            var layers = model.ToPython().GetAttr("layers");

            int i = 0;
            foreach (var layer in layers)
            {
                (layer as PyObject).SetAttr("trainable", bval);

                i++;
                if (count > 0 && i >= count)
                    break;
            }
        }
    }
}
